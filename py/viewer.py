"""Interactive viewer (PyQt5): the DRAM traffic of the whole pipeline through an L2 model (from the
tile orders and row loops of the kernels), the read/write dependency graph below a start level
(Graphviz), its L2 traffic, the device memory layout of the BFS buffers and the tile order of the
mma leaf GEMM.

    python py/viewer.py

Replaces the C++ experimental/schedule_viz/l2_viewer. Needs PyQt5, numpy, opencv-python-headless;
the "Dependency graph" tab also needs Graphviz (see gemm_l2/graphviz.py).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QByteArray, Qt, QTimer
from PyQt5.QtGui import QIcon, QImage, QPainter, QPalette, QPixmap
from PyQt5.QtSvg import QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout,
                             QWidget)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemm_l2 import deps, encode, encode_l2, graphviz, grid, l2_sim, tile_order  # noqa: E402
from gemm_l2.layout import layout  # noqa: E402
from gemm_l2.images import validate  # noqa: E402
from encode_view import EncodeView, info_html, l2_html, passes_html, schedule_html  # noqa: E402
from layout_view import LayoutView, operands_html, reuse_html, summary_html  # noqa: E402
from overlap_view import OverlapView, overlap_html  # noqa: E402
from gemm_l2.overlap import timeline as overlap_timeline  # noqa: E402
from gemm_l2.reuse import analyse, group_choices, place  # noqa: E402

MB = 1024 * 1024
ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_icon.svg")
# (label, l2_sim.Gpu); L2 size, SMs and bandwidth as far as known (gemm_l2/l2_sim.py)
PRESETS = (("Custom", None),) + tuple((f"{g.label} ({g.l2_mb:g} MB)", g) for g in l2_sim.GPUS)
LOOP_NAMES = ("encode", "bilinear", "reconstruct", "decode")


def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QScrollArea):
    """Scrollable, zoomable image"""

    def __init__(self):
        super().__init__()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setWidget(self.label)
        self.setBackgroundRole(QPalette.Dark)
        self.image = None
        self.pixmap = QPixmap()
        self.zoom = 1.0
        self.fit_width = False

    def set_image(self, image):
        self.image = image
        self.pixmap = QPixmap.fromImage(to_qimage(image))
        self.refresh()

    def set_message(self, text):
        self.image = None
        self.pixmap = QPixmap()
        self.label.setPixmap(QPixmap())
        self.label.setText(text)
        self.label.adjustSize()

    def set_zoom(self, zoom, fit_width):
        self.zoom, self.fit_width = zoom, fit_width
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_width:
            self.refresh()

    def refresh(self):
        if self.pixmap.isNull():
            return
        zoom = min(1.0, (self.viewport().width() - 2) / self.pixmap.width()) if self.fit_width else self.zoom
        pm = self.pixmap if zoom == 1.0 else self.pixmap.scaled(max(1, int(self.pixmap.width() * zoom)),
                                                                max(1, int(self.pixmap.height() * zoom)),
                                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setText("")
        self.label.setPixmap(pm)
        self.label.adjustSize()


class SvgView(QGraphicsView):
    """SVG view: wheel zooms, drag pans"""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.renderer = None

    def set_svg(self, data: bytes):
        self.scene().clear()
        self.renderer = QSvgRenderer(QByteArray(data))
        item = QGraphicsSvgItem()
        item.setSharedRenderer(self.renderer)
        self.scene().addItem(item)
        self.scene().setSceneRect(item.boundingRect())
        self.resetTransform()
        self.fitInView(item, Qt.KeepAspectRatio)

    def set_message(self, text):
        self.scene().clear()
        self.scene().addText(text)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viewer")
        self.setWindowIcon(QIcon(ICON))

        # ---- controls
        self.preset = QComboBox()
        for name, _ in PRESETS:
            self.preset.addItem(name)
        self.l2 = QDoubleSpinBox()
        self.l2.setRange(1, 1024)
        self.l2.setDecimals(1)
        self.l2.setSuffix(" MB")
        self.l2.setValue(72)
        self.N = QComboBox()
        self.N.setEditable(True)
        for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
            self.N.addItem(str(n))
        self.N.setCurrentText("8192")
        self.R = QSpinBox()
        self.R.setRange(1, 5)
        self.R.setValue(3)
        self.R.setToolTip("Recursions: 7^R leaf multiplications of (N/2^R)^2 blocks (GPU code: R <= 4)")

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.auto = QCheckBox("Update automatically")
        self.auto.setChecked(True)
        update = QPushButton("Update")
        save = QPushButton("Save...")

        form = QFormLayout()
        form.addRow("GPU", self.preset)
        form.addRow("L2 size", self.l2)
        form.addRow("N", self.N)
        form.addRow("R (recursions)", self.R)
        controls = QVBoxLayout()
        controls.addLayout(form)
        controls.addWidget(self.info)
        controls.addWidget(self.auto)
        controls.addWidget(update)
        controls.addWidget(save)
        controls.addSpacing(12)
        controls.addWidget(self.stats)
        controls.addStretch()
        panel = QWidget()
        panel.setLayout(controls)
        panel.setFixedWidth(310)

        # ---- views
        self.tabs = QTabWidget()
        self.sim_tab = self._sim_tab()
        self.graph_tab, self.graph_view = self._graph_tab()
        self.traffic_tab = self._traffic_tab()
        self.layout_tab = self._layout_tab()
        self.tile_tab = self._tile_tab()
        self.encode_tab = self._encode_tab()
        self.overlap_tab = self._overlap_tab()
        self.tabs.addTab(self.sim_tab, "L2 simulation")
        self.tabs.addTab(self.graph_tab, "Dependency graph")
        self.tabs.addTab(self.traffic_tab, "L2 traffic")
        self.tabs.addTab(self.layout_tab, "Memory layout")
        self.tabs.addTab(self.tile_tab, "Tile order")
        self.tabs.addTab(self.encode_tab, "Encoder")
        self.tabs.addTab(self.overlap_tab, "Early products")

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(panel)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.resize(1500, 950)

        # ---- behaviour
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.recompute)
        self.valid = set()  # tabs computed for the current parameters
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.l2.valueChanged.connect(self._l2_changed)
        for signal in (self.N.currentTextChanged, self.R.valueChanged):
            signal.connect(self._changed)
        update.clicked.connect(self.recompute)
        self.tabs.currentChanged.connect(lambda _: self.ensure_tab())
        save.clicked.connect(self.save)
        self.preset.setCurrentIndex(1)
        self.update_info()
        self.recompute()

    # ---- parameters
    def n(self):
        try:
            return int(self.N.currentText())
        except ValueError:
            return 0

    def r(self):
        return self.R.value()

    def l2_bytes(self):
        return int(self.l2.value() * MB)

    def leaf_blocks(self):
        M = self.n() >> self.r()
        return self.l2_bytes() // (M * M * 2) if M else 0

    # ---- tabs
    def _sim_tab(self):
        self.sim_leaf = QComboBox()
        self.sim_leaf.addItem("mma leaf GEMM", "mma")
        self.sim_leaf.addItem("fused last level", "fused")
        self.sim_leaf.setToolTip("caps(): forward, batched mma leaf GEMM, backward; or the fused pipeline: "
                                 "fused_forward_pass, batched_fused, fused_backward_pass")
        self.sim_config = QComboBox()
        self.sim_order = QComboBox()
        for o in tile_order.ORDERS:
            self.sim_order.addItem(f"{o}: {tile_order.order_name(o)}", o)
        self.sim_order.setToolTip("Tile order of the GEMM stage (mma_gemm_t::set_order(), the fused order argument)")
        self.sim_sms = QSpinBox()
        self.sim_sms.setRange(1, 512)
        self.sim_sms.setValue(128)
        self.sim_sms.setToolTip("SMs: the blocks of one wave are SMs x resident blocks per SM")
        self.sim_bw = QSpinBox()
        self.sim_bw.setRange(10, 20000)
        self.sim_bw.setSuffix(" GB/s")
        self.sim_bw.setValue(1008)
        self.sim_bw.setToolTip("DRAM bandwidth: time = DRAM bytes / bandwidth (a lower bound per stage)")
        self.sim_loops = []
        for name in LOOP_NAMES:
            box = QComboBox()
            for v in (1 << i for i in range(13)):
                box.addItem(str(v), v)
            box.setToolTip(f"{name} loops (GEMM_TUNED_BFS): rows per loop = rows / loops; at most the leaf size")
            self.sim_loops.append(box)
        tuned = QPushButton("Tuned")
        tuned.setToolTip("Loop counts and mma leaf configuration of gen/tuning_table.hpp for the GPU preset "
                         "(set automatically when N, R or the GPU change)")
        self.sim_basis = QCheckBox("Alternative basis")
        self.sim_basis.setChecked(True)
        self.sim_basis.setToolTip("Include the encode and decode passes")
        self.sim_tuned_info = QLabel()
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        for label, widget in (("Leaf", self.sim_leaf), ("Config", self.sim_config), ("Order", self.sim_order),
                              ("SMs", self.sim_sms), ("DRAM", self.sim_bw)):
            row1.addWidget(QLabel(label))
            row1.addWidget(widget)
        row1.addStretch()
        row2.addWidget(QLabel("Loops"))
        for name, box in zip(LOOP_NAMES, self.sim_loops):
            row2.addWidget(QLabel(name))
            row2.addWidget(box)
        row2.addWidget(tuned)
        row2.addWidget(self.sim_basis)
        row2.addWidget(self.sim_tuned_info, 1)
        self.sim_view = QTextBrowser()
        box = QVBoxLayout()
        box.addLayout(row1)
        box.addLayout(row2)
        box.addWidget(self.sim_view, 1)
        tab = QWidget()
        tab.setLayout(box)
        self._fill_sim_configs()
        self.sim_leaf.currentIndexChanged.connect(lambda *_: (self._fill_sim_configs(), self._apply_tuned(),
                                                              self._sim_changed()))
        for signal in ([self.sim_config.currentIndexChanged, self.sim_order.currentIndexChanged,
                        self.sim_sms.valueChanged, self.sim_bw.valueChanged, self.sim_basis.toggled]
                       + [b.currentIndexChanged for b in self.sim_loops]):
            signal.connect(lambda *_: self._sim_changed())
        tuned.clicked.connect(lambda: (self._apply_tuned(), self._sim_changed()))
        return tab

    def _fill_sim_configs(self):
        self.sim_config.blockSignals(True)
        self.sim_config.clear()
        try:
            configs = tile_order.CONFIGS if self.sim_leaf.currentData() == "mma" else l2_sim.fused_configs()
        except (OSError, ValueError) as e:  # fused configurations are read from src/fused_strassen.cu
            configs = []
            self.sim_tuned_info.setText(str(e))
        for i, c in enumerate(configs):
            self.sim_config.addItem(f"#{i} {c.BM}x{c.BN}, k {c.BK}, {c.stages} stages, {c.blocks_per_sm} per SM")
        if self.sim_leaf.currentData() == "fused" and self.sim_config.count() > 8:
            self.sim_config.setCurrentIndex(8)
        self.sim_config.blockSignals(False)

    def _gpu(self):
        return PRESETS[self.preset.currentIndex()][1]

    def _apply_tuned(self):
        """Loop counts (and the mma leaf configuration) of gen/tuning_table.hpp for the preset GPU"""
        gpu, N, R = self._gpu(), self.n(), self.r()
        if not N or R < 1:
            return
        t = l2_sim.tuned(gpu.name if gpu else "default", N, R)
        widgets = self.sim_loops + [self.sim_config, self.sim_order]
        for w in widgets:
            w.blockSignals(True)
        for box, v in zip(self.sim_loops, t["loops"]):
            i = box.findData(min(v, N >> R))
            if i >= 0:
                box.setCurrentIndex(i)
        enc_loops = t["loops"][3 if self.enc_op.currentText() == "decode C" else 0]
        self.enc_loops.blockSignals(True)
        self.enc_loops.setCurrentIndex(max(0, self.enc_loops.findData(min(enc_loops, N >> R))))
        self.enc_loops.blockSignals(False)
        text = f"loops: {t['loops_from']}"
        if self.sim_leaf.currentData() == "mma":
            config, order = t["mma"]
            self.sim_config.setCurrentIndex(config)
            self.sim_order.setCurrentIndex(max(0, self.sim_order.findData(order)))
            text += f", leaf: {t['mma_from']}"
        for w in widgets:
            w.blockSignals(False)
        self.sim_tuned_info.setText(f"<span style='color:#777'>tuning table ({text})</span>")

    def _sim_changed(self):
        self.valid.discard("sim")
        self.ensure_tab()

    def _graph_tab(self):
        self.engine = QComboBox()
        self.engine.addItems(graphviz.ENGINES)
        self.rankdir = QComboBox()
        self.rankdir.addItems(["LR", "TB"])
        self.graph_info = QLabel()
        save_dot = QPushButton("Save .dot / .svg...")
        top = QHBoxLayout()
        for label, widget in (("Engine", self.engine), ("Direction", self.rankdir)):
            top.addWidget(QLabel(label))
            top.addWidget(widget)
        top.addWidget(save_dot)
        top.addWidget(self.graph_info, 1)
        view = SvgView()
        box = QVBoxLayout()
        box.addLayout(top)
        box.addWidget(view, 1)
        tab = QWidget()
        tab.setLayout(box)
        for signal in (self.engine.currentIndexChanged, self.rankdir.currentIndexChanged):
            signal.connect(lambda *_: self._graph_changed())
        save_dot.clicked.connect(self.save_graph)
        self.dot_text, self.svg = "", b""
        return tab, view

    def _traffic_tab(self):
        self.traffic_depth = QSpinBox()
        self.traffic_depth.setRange(1, 2)
        self.traffic = QTextBrowser()
        top = QHBoxLayout()
        label = QLabel("Subtree levels")
        label.setToolTip("Levels of the subtrees run as a group (forward, leaf GEMMs, reconstruct): "
                         "1 = one parent and its 7 leaf products; the subtrees start at level R - this")
        top.addWidget(label)
        top.addWidget(self.traffic_depth)
        top.addStretch()
        box = QVBoxLayout()
        box.addLayout(top)
        box.addWidget(self.traffic, 1)
        tab = QWidget()
        tab.setLayout(box)
        self.traffic_depth.valueChanged.connect(lambda *_: (self.valid.discard("traffic"), self.ensure_tab()))
        return tab

    def _layout_tab(self):
        self.columns = QSpinBox()
        self.columns.setRange(0, 64)
        self.columns.setSpecialValueText("auto")
        self.columns.setToolTip("Columns the S buffer is folded into (it is one address range, row after row)")
        self.show_inputs = QCheckBox("A and B")
        self.show_inputs.setChecked(True)
        self.show_sums = QCheckBox("S1..S6")
        self.show_sums.setChecked(True)
        self.reuse_mode = QComboBox()
        self.reuse_mode.addItems(["Products: own buffers", "Mark blocks for products", "Products in A, B, S",
                                  "Leaf operands"])
        self.reuse_mode.setToolTip("Which leaf x leaf blocks of A, B and S could hold the leaf products (gemm_l2/reuse.py): "
                                   "mark them, or place the products there and see the memory saved")
        self.reuse_groups = QComboBox()
        self.reuse_groups.setToolTip("Leaf GEMMs as consecutive batches over the level R-1 nodes (1 = one batched "
                                     "GEMM, as today); a cell frees up after the last group that reads it")
        fit = QPushButton("Fit")
        fit.setToolTip("Zoom 100%: the whole layout (or its width) fits the view")
        self.layout_info = QLabel()
        self.layout_info.setWordWrap(True)
        self.layout_zoom, self.layout_zoom_label, self.layout_fit = self._zoom_controls(10, 5000)
        self.layout_zoom.setToolTip("100% = the layout fits the view; the mouse wheel zooms too")
        self.layout_fit.setToolTip("Fit the width of the layout instead of the whole layout")
        top = QHBoxLayout()
        top.addWidget(QLabel("Columns"))
        top.addWidget(self.columns)
        top.addWidget(self.show_inputs)
        top.addWidget(self.show_sums)
        top.addWidget(self.reuse_mode)
        top.addWidget(QLabel("GEMM groups"))
        top.addWidget(self.reuse_groups)
        top.addWidget(QLabel("Zoom"))
        top.addWidget(self.layout_zoom)
        top.addWidget(self.layout_zoom_label)
        top.addWidget(self.layout_fit)
        top.addWidget(fit)
        top.addWidget(self.layout_info, 1)
        self.layout_view = LayoutView()
        self.layout_view.on_select = self.layout_info.setText
        self.layout_view.on_zoom = self._layout_zoomed
        for signal in (self.layout_zoom.valueChanged, self.layout_fit.toggled):
            signal.connect(lambda *_: self._layout_zoom_changed())
        box = QVBoxLayout()
        box.addLayout(top)
        box.addWidget(self.layout_view, 1)
        tab = QWidget()
        tab.setLayout(box)
        for signal in (self.columns.valueChanged, self.show_inputs.toggled, self.show_sums.toggled,
                       self.reuse_mode.currentIndexChanged, self.reuse_groups.activated):
            signal.connect(lambda *_: (self.valid.discard("layout"), self.ensure_tab()))
        fit.clicked.connect(self.layout_view.fit)
        return tab

    def _overlap_tab(self):
        self.ov_tflops = QDoubleSpinBox()
        self.ov_tflops.setRange(1, 5000)
        self.ov_tflops.setDecimals(0)
        self.ov_tflops.setValue(300)
        self.ov_tflops.setSuffix(" TFLOPS")
        self.ov_tflops.setToolTip("Effective leaf GEMM throughput (RTX 4090, half accumulation: about 300)")
        self.ov_eff = QSpinBox()
        self.ov_eff.setRange(10, 100)
        self.ov_eff.setValue(90)
        self.ov_eff.setSuffix(" %")
        self.ov_eff.setToolTip("Share of the GPU preset's DRAM bandwidth the sums kernels reach (4090: about 90%)")
        self.ov_bw = QDoubleSpinBox()
        self.ov_bw.setRange(10, 20000)
        self.ov_bw.setDecimals(0)
        self.ov_bw.setValue(1008)
        self.ov_bw.setSuffix(" GB/s")
        self.ov_bw.setToolTip("DRAM bandwidth (from the GPU preset)")
        self.ov_paths = QCheckBox("Paths")
        self.ov_paths.setToolTip("Label every product with its path m.m.m from the root (child numbers)")
        fit = QPushButton("Fit")
        top = QHBoxLayout()
        for label, widget in (("Leaf GEMM", self.ov_tflops), ("DRAM", self.ov_bw), ("of it", self.ov_eff)):
            top.addWidget(QLabel(label))
            top.addWidget(widget)
        top.addWidget(self.ov_paths)
        top.addWidget(fit)
        self.ov_info = QLabel("Click a product to see the blocks of A, B and the sums it reads.")
        self.ov_info.setWordWrap(True)
        top.addWidget(self.ov_info, 1)
        self.overlap_view = OverlapView()
        self.overlap_view.on_select = self.ov_info.setText
        box = QVBoxLayout()
        box.addLayout(top)
        box.addWidget(self.overlap_view, 1)
        tab = QWidget()
        tab.setLayout(box)
        for signal in (self.ov_tflops.valueChanged, self.ov_eff.valueChanged, self.ov_bw.valueChanged,
                       self.ov_paths.toggled):
            signal.connect(lambda *_: (self.valid.discard("overlap"), self.ensure_tab()))
        fit.clicked.connect(self.overlap_view.fit)
        return tab

    def compute_overlap(self):
        N, R = self.n(), self.r()
        gpu = self._gpu()
        if gpu and self.ov_bw.value() != gpu.dram_gbs:
            self.ov_bw.blockSignals(True)
            self.ov_bw.setValue(gpu.dram_gbs)
            self.ov_bw.blockSignals(False)
        tl = overlap_timeline(N, R, self.ov_tflops.value(), self.ov_bw.value() * self.ov_eff.value() / 100.0)
        self.overlap_view.show(N, R, tl, self.ov_paths.isChecked() and 7 ** R <= 400)
        self.stats.setText(overlap_html(N, R, tl))
        self.statusBar().showMessage("Click a product for its operands in A, B and the sums (or a block for its product); hover for details; wheel: zoom, drag: pan")

    def _encode_tab(self):
        self.enc_mode = QComboBox()
        self.enc_mode.addItems(["Levels", "Schedule", "L2"])
        self.enc_mode.setToolTip("Levels: what every level computes; Schedule: how the kernel runs it, step by step "
                                 "(row loops, levels per grid barrier, grid rounds); L2: what is in L2 and what comes "
                                 "from DRAM in every iteration of the original implementation")
        self.enc_trace = None
        self.enc_step = QSlider(Qt.Horizontal)
        self.enc_step.setFixedWidth(220)
        self.enc_step.setToolTip("Grid-barrier step of the kernel")
        self.enc_step_label = QLabel("step 1")
        self.enc_step_label.setMinimumWidth(80)
        self.enc_pipelined = QCheckBox("Pipelined")
        self.enc_pipelined.setChecked(True)
        self.enc_pipelined.setToolTip("PIPELINED_LEVELS=ON: the levels overlap across row loops (one step runs level l "
                                      "on loop s - (l - 1)); off: the original loop x level order")
        self.enc_hints = QCheckBox("Last-use hints")
        self.enc_hints.setToolTip("The last access of every unit (rows the pass never touches again) as __ldlu (last read) "
                                  "or __stcs (last write): the line leaves L2 first, before data still needed")
        self.enc_comparison = None
        self.enc_op = QComboBox()
        self.enc_op.addItems(encode.OPERATIONS)
        self.enc_op.setToolTip("Encoder on A or B (one kernel transforms both), decoder on the root's C")
        self.enc_level = QSlider(Qt.Horizontal)
        self.enc_level.setRange(0, 3)
        self.enc_level.setFixedWidth(160)
        self.enc_level.setToolTip("0: the input; l: the blocks level l reads and writes, contents after it")
        self.enc_level_label = QLabel("input")
        self.enc_level_label.setMinimumWidth(70)
        self.enc_loops = QComboBox()
        for v in (1 << i for i in range(13)):
            self.enc_loops.addItem(str(v), v)
        self.enc_loops.setToolTip("Row loops (GEMM_TUNED_BFS encode / decode); set from the tuning table")
        self.enc_loop = QSpinBox()
        self.enc_loop.setToolTip("The loop whose rows are shown")
        self.enc_stripes = QCheckBox("Rows of the loop")
        self.enc_stripes.setChecked(True)
        self.enc_texts = QCheckBox("Expressions")
        self.enc_texts.setChecked(True)
        self.enc_texts.setToolTip("Write what every block holds into it (R <= 2); hover always shows it")
        self.enc_blocks = QSpinBox()
        self.enc_blocks.setRange(1, 32)
        self.enc_blocks.setValue(12)
        self.enc_blocks.setToolTip("Resident bfs_kernel blocks per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor, "
                                   "printed as 'Number of feasible active blocks per SM' with REPORT_STATISTICS); "
                                   "12 = 48 warps on sm_86 / sm_89. Grid = SMs x this / 4")
        fit = QPushButton("Fit")
        top = QHBoxLayout()
        self.enc_level_caption, self.enc_step_caption = QLabel("Level"), QLabel("Step")
        for label, widget in ((QLabel("View"), self.enc_mode), (QLabel("Transform"), self.enc_op),
                              (self.enc_level_caption, self.enc_level)):
            top.addWidget(label)
            top.addWidget(widget)
        top.addWidget(self.enc_level_label)
        top.addWidget(self.enc_step_caption)
        top.addWidget(self.enc_step)
        top.addWidget(self.enc_step_label)
        top.addWidget(self.enc_pipelined)
        top.addWidget(self.enc_hints)
        self.enc_loop_caption = QLabel("Loop")
        for label, widget in ((QLabel("Loops"), self.enc_loops), (self.enc_loop_caption, self.enc_loop),
                              (QLabel("Blocks/SM"), self.enc_blocks)):
            top.addWidget(label)
            top.addWidget(widget)
        top.addWidget(self.enc_stripes)
        top.addWidget(self.enc_texts)
        top.addWidget(fit)
        top.addStretch()
        self.encode_view = EncodeView()
        self.enc_info = QTextBrowser()
        self.enc_info.setFixedWidth(480)
        body = QHBoxLayout()
        body.addWidget(self.encode_view, 1)
        body.addWidget(self.enc_info)
        box = QVBoxLayout()
        box.addLayout(top)
        box.addLayout(body, 1)
        tab = QWidget()
        tab.setLayout(box)
        self.enc_states = {}  # (R, operation) -> encode.expressions()
        self.enc_level.valueChanged.connect(lambda *_: self._encode_changed(False))
        self.enc_step.valueChanged.connect(lambda *_: self._encode_changed(False))
        self.enc_mode.currentIndexChanged.connect(lambda *_: self._encode_changed(True))
        self.enc_pipelined.toggled.connect(lambda *_: self._encode_changed(False))
        self.enc_hints.toggled.connect(lambda *_: self._encode_changed(False))
        self.enc_op.currentIndexChanged.connect(lambda *_: (self._apply_tuned(), self._encode_changed(True)))
        for signal in (self.enc_loops.currentIndexChanged, self.enc_loop.valueChanged, self.enc_stripes.toggled,
                       self.enc_texts.toggled, self.enc_blocks.valueChanged):
            signal.connect(lambda *_: self._encode_changed(False))
        fit.clicked.connect(self.encode_view.fit)
        return tab

    def _encode_changed(self, refit):
        if refit:
            self.encode_view.fitted = False
        self.valid.discard("encode")
        self.ensure_tab()

    def _tile_tab(self):
        self.tile_config = QComboBox()
        for i, c in enumerate(tile_order.CONFIGS):
            self.tile_config.addItem(f"#{i} {c.name}")
        self.tile_config.setToolTip("Configuration of src/mma_gemm.cu (mma_gemm_t::config_name())")
        self.tile_orders = QComboBox()
        for o in tile_order.ORDERS:
            self.tile_orders.addItem(f"{o}: {tile_order.order_name(o)}", o)
        self.tile_orders.setToolTip("mma_gemm_t::set_order() / GEMM_TUNED_LEAF order")
        self.tile_sms = QSpinBox()
        self.tile_sms.setRange(1, 256)
        self.tile_sms.setValue(58)
        self.tile_sms.setToolTip("SMs of the GPU (RTX 3080 Ti Laptop: 58, RTX 4090: 128): blocks of one wave")
        self.tile_info = QLabel()
        self.tile_zoom, self.tile_zoom_label, self.tile_fit = self._zoom_controls(10, 200)
        self.tile_fit.setToolTip("Scale the image to the width of the view")
        top = QHBoxLayout()
        for label, widget in (("Config", self.tile_config), ("Order", self.tile_orders), ("SMs", self.tile_sms),
                              ("Zoom", self.tile_zoom)):
            top.addWidget(QLabel(label))
            top.addWidget(widget)
        top.addWidget(self.tile_zoom_label)
        top.addWidget(self.tile_fit)
        top.addWidget(self.tile_info, 1)
        self.tile_view = ImageView()
        for signal in (self.tile_zoom.valueChanged, self.tile_fit.toggled):
            signal.connect(lambda *_: self._tile_zoom_changed())
        box = QVBoxLayout()
        box.addLayout(top)
        box.addWidget(self.tile_view, 1)
        tab = QWidget()
        tab.setLayout(box)
        for signal in (self.tile_config.currentIndexChanged, self.tile_orders.currentIndexChanged,
                       self.tile_sms.valueChanged):
            signal.connect(lambda *_: (self.valid.discard("tiles"), self.ensure_tab()))
        return tab

    # ---- events
    def _preset_changed(self, i):
        gpu = PRESETS[i][1]
        if gpu:
            for widget, value in ((self.l2, gpu.l2_mb), (self.tile_sms, gpu.sms), (self.sim_sms, gpu.sms),
                                  (self.sim_bw, int(gpu.dram_gbs))):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self._changed()

    def _l2_changed(self, value):
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(next((i for i, p in enumerate(PRESETS) if p[1] and p[1].l2_mb == value), 0))
        self.preset.blockSignals(False)
        self._changed()

    def _changed(self, *_):
        self.update_info()
        self.encode_view.fitted = False
        self._apply_tuned()
        self.valid.clear()
        if self.auto.isChecked():
            self.timer.start()

    def _graph_changed(self):
        self.valid.discard("graph")
        self.ensure_tab()

    @staticmethod
    def _zoom_controls(low, high):
        """Zoom slider (percent), its label and a Fit width check box, for a tab's top row"""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(100)
        slider.setFixedWidth(160)
        label = QLabel("100%")
        label.setMinimumWidth(48)
        return slider, label, QCheckBox("Fit width")

    def _tile_zoom_changed(self):
        self.tile_zoom_label.setText(f"{self.tile_zoom.value()}%")
        self.tile_zoom.setEnabled(not self.tile_fit.isChecked())
        self.tile_view.set_zoom(self.tile_zoom.value() / 100.0, self.tile_fit.isChecked())

    def _layout_zoom_changed(self):
        self.layout_zoom_label.setText(f"{self.layout_zoom.value()}%")
        self.layout_view.set_zoom(self.layout_zoom.value() / 100.0, self.layout_fit.isChecked())

    def _layout_zoomed(self, zoom):
        """the wheel or Fit changed the layout's zoom: show it on the slider"""
        self.layout_zoom.blockSignals(True)
        self.layout_zoom.setValue(round(zoom * 100))
        self.layout_zoom.blockSignals(False)
        self.layout_zoom_label.setText(f"{round(zoom * 100)}%")

    def update_info(self):
        error = validate(self.n(), self.r(), 0)
        if error:
            self.info.setText(f"<span style='color:#c00'>{error}</span>")
            return
        M = self.n() >> self.r()
        self.info.setText(f"Leaf: {M} x {M} halves ({M * M * 2 / MB:.2f} MB), {7 ** self.r()} leaf multiplications"
                          f"<br>L2 holds {self.leaf_blocks()} leaf blocks")

    # ---- computation
    def recompute(self):
        self.timer.stop()
        self.valid.clear()
        self.stats.setText("")
        error = self._error()
        if error:
            self.statusBar().showMessage(error)
            return
        self.ensure_tab()

    def _error(self):
        """Why the current tab cannot be computed; the L2 size limit only matters for the L2 traffic tab"""
        l2_free = self.tabs.currentWidget() in (self.layout_tab, self.tile_tab, self.graph_tab, self.sim_tab,
                                                self.encode_tab, self.overlap_tab)
        return validate(self.n(), self.r(), 0 if l2_free else self.l2_bytes())

    def ensure_tab(self):
        """Compute the visible tab if its parameters changed (tabs are computed lazily)."""
        if self._error():
            return
        tab = {self.sim_tab: "sim", self.graph_tab: "graph", self.traffic_tab: "traffic",
               self.layout_tab: "layout", self.tile_tab: "tiles",
               self.encode_tab: "encode", self.overlap_tab: "overlap"}[self.tabs.currentWidget()]
        if tab in self.valid:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Computing...")
        QApplication.processEvents()
        try:
            getattr(self, "compute_" + tab)()
            self.valid.add(tab)
            if self.statusBar().currentMessage() == "Computing...":
                self.statusBar().showMessage("Ready")
        except Exception as e:  # noqa: BLE001 - show any failure in the view
            self.statusBar().showMessage(f"Failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def compute_sim(self):
        N, R, l2 = self.n(), self.r(), self.l2_bytes()
        leaf, config, order = self.sim_leaf.currentData(), self.sim_config.currentIndex(), self.sim_order.currentData()
        sms, bw = self.sim_sms.value(), self.sim_bw.value()
        loops = tuple(b.currentData() for b in self.sim_loops)
        M = N >> R
        too_many = [f"{name} {v}" for name, v in zip(LOOP_NAMES, loops) if v > M]
        if too_many:
            self.sim_view.setHtml(f"Loop counts larger than the leaf size {M}: {', '.join(too_many)}")
            return
        if config < 0:
            self.sim_view.setHtml("No configuration (see the message next to the loop counts).")
            return
        try:
            stages = l2_sim.pipeline(N, R, l2, sms, loops, leaf, config, order, config, order,
                                     self.sim_basis.isChecked())
            orders = l2_sim.compare_orders(N, R, l2, sms, leaf, config, tile_order.ORDERS)
        except ValueError as e:
            self.sim_view.setHtml(str(e))
            return
        self.sim_view.setHtml(sim_html(N, R, l2, bw, leaf, config, order, loops, stages, orders))
        total = sum(s.total for s in stages)
        self.statusBar().showMessage(f"N={N}, R={R}, L2 {self.l2.value():g} MB: {total / MB:,.0f} MB DRAM traffic, "
                                     f"{sum(s.ms(bw) for s in stages):.2f} ms at {bw} GB/s")

    def compute_graph(self):
        """the whole tree of R levels: forward levels, the 7^R leaf GEMMs, reconstruct"""
        depth = self.r()
        if depth > 3:
            self.dot_text, self.svg = "", b""
            self.graph_view.set_message(f"R = {depth}: {7 ** depth} leaf products, too large to draw (R <= 3).")
            self.graph_info.setText("")
            return
        ops = deps.subtree_ops(depth)
        self.dot_text = deps.to_dot(ops, depth, self.rankdir.currentText())
        s = deps.summary(ops)
        self.graph_info.setText(f"{s['ops']} ops ({s['mults']} mults), {s['blocks']} blocks, {s['block_reads']} "
                                f"reads, {s['block_writes']} writes; reads per block {s['reads_per_block']}")
        try:
            self.svg = graphviz.render(self.dot_text, self.engine.currentText())
            self.graph_view.set_svg(self.svg)
            self.statusBar().showMessage("Wheel: zoom, drag: pan")
        except RuntimeError as e:
            self.svg = b""
            self.graph_view.set_message(str(e))

    def compute_traffic(self):
        depth = self.traffic_depth.value()
        R, cache = self.r(), self.leaf_blocks()
        if R <= depth:
            self.traffic.setHtml(f"R must be larger than the subtree levels ({depth}).")
            return
        subtrees = 7 ** (R - depth)
        one = deps.summary(deps.subtree_ops(depth))["blocks"]
        groups = sorted({1, 2, 3, 4, 6, 8, 16, subtrees} | {max(1, cache // one)})
        rows = ""
        for g in (g for g in groups if g <= subtrees):
            cells = ""
            for ring, discard in ((False, False), (True, False), (True, True)):
                t = deps.simulate(R, depth, cache, g, ring, discard)
                cells += f"<td align='right'>{t.total / subtrees:.1f}</td>"
            label = f"{g} (today: whole level)" if g == subtrees else str(g)
            rows += f"<tr><td>{label}</td><td align='right'>{t.peak_group_blocks}</td>{cells}</tr>"
        self.traffic.setHtml(
            f"<h3>DRAM traffic per subtree (M x M blocks), R={R}, subtrees of {depth} levels, L2 = {cache} blocks</h3>"
            f"<p>{subtrees} subtrees below level {R - depth}; one subtree touches {one} blocks. Grouped execution: "
            "forward, leaf GEMMs and reconstruct of <i>group</i> subtrees, then the next group. LRU write-back "
            "model (gemm_l2/deps.py). <i>ring</i>: S and product buffers reused by the next group; "
            "<i>discard</i>: dirty data dropped after its last read.</p>"
            "<table border='1' cellspacing='0' cellpadding='4'><tr><th>group</th><th>blocks per group</th>"
            "<th>plain</th><th>ring</th><th>ring + discard</th></tr>" + rows + "</table>")

    def compute_layout(self):
        lay = layout(self.n(), self.r())
        choices = [str(g) for g in group_choices(self.r())]
        if [self.reuse_groups.itemText(i) for i in range(self.reuse_groups.count())] != choices:
            keep = self.reuse_groups.currentText()
            self.reuse_groups.blockSignals(True)
            self.reuse_groups.clear()
            self.reuse_groups.addItems(choices)
            self.reuse_groups.setCurrentIndex(choices.index(keep) if keep in choices else 0)
            self.reuse_groups.blockSignals(False)
        mode = self.reuse_mode.currentIndex()
        self.reuse_groups.setEnabled(mode in (1, 2))
        res = None
        if mode > 0:
            res = analyse(lay, int(self.reuse_groups.currentText() or 1) if mode < 3 else 1)
            if mode == 2:
                place(res)
        self.layout_view.show_layout(lay, self.l2_bytes(), self.columns.value(), self.show_inputs.isChecked(),
                                     self.show_sums.isChecked(), res, mode == 3)
        self.layout_info.setText("Hover a block for its address range, click a node for its parent and children.")
        panel = operands_html(res) if mode == 3 else reuse_html(res) if res is not None else ""
        self.stats.setText(panel + summary_html(lay, self.l2_bytes()))
        self.statusBar().showMessage("Wheel: zoom, drag: pan")

    def compute_encode(self):
        N, R, op = self.n(), self.r(), self.enc_op.currentText()
        mode = self.enc_mode.currentText()
        for w in (self.enc_level_caption, self.enc_level, self.enc_level_label, self.enc_stripes, self.enc_texts,
                  self.enc_loop, self.enc_loop_caption):
            w.setVisible(mode == "Levels")
        for w in (self.enc_step_caption, self.enc_step, self.enc_step_label):
            w.setVisible(mode != "Levels")
        self.enc_pipelined.setVisible(mode == "Schedule")
        self.enc_hints.setVisible(mode == "L2")
        self.enc_step_caption.setText("Iteration" if mode == "L2" else "Step")
        if mode == "Schedule":
            return self._compute_schedule(N, R, op)
        if mode == "L2":
            return self._compute_l2(N, R, op)
        for w in (self.enc_level, self.enc_loop):
            w.blockSignals(True)
        self.enc_level.setMaximum(R)
        loops = min(self.enc_loops.currentData(), N >> R)
        self.enc_loop.setRange(0, loops - 1)
        for w in (self.enc_level, self.enc_loop):
            w.blockSignals(False)
        level, loop = self.enc_level.value(), self.enc_loop.value()
        self.enc_level_label.setText(f"level {level}" if level else "input")
        if (R, op) not in self.enc_states:
            self.enc_states[(R, op)] = encode.expressions(R, op)
        self.enc_texts.setEnabled(R <= 2)
        G = grid.grid_blocks(self.sim_sms.value(), self.enc_blocks.value())
        its = grid.iterations(N, R, "decode" if op == "decode C" else "encode", loops, G)
        self.encode_view.show_transform(N, R, op, level, loops, loop, self.enc_stripes.isChecked(),
                                        self.enc_texts.isChecked(), self.enc_states[(R, op)], its)
        _, levels = l2_sim.tree(N, R)
        root = levels[0][0]
        if op == "decode C":
            stage = l2_sim.simulate_pass("decode", l2_sim.decode_levels(N, R, root.C), loops, self.l2_bytes())
        else:
            stage = l2_sim.simulate_pass("encode", l2_sim.encode_levels(N, R, [root.L, root.R]), loops,
                                         self.l2_bytes())
        sim_loops = tuple(min(b.currentData(), N >> R) for b in self.sim_loops)
        self.enc_info.setHtml(info_html(N, R, op, level, loops, loop, self.l2_bytes(), stage)
                              + passes_html(N, R, sim_loops, G))
        self.statusBar().showMessage("Hover a block for its contents; wheel: zoom, drag: pan")

    def _compute_l2(self, N, R, op):
        loops = min(self.enc_loops.currentData(), N >> R)
        kind = "decode" if op == "decode C" else "encode"
        hints = self.enc_hints.isChecked()
        key = (N, R, kind, loops, self.l2_bytes(), hints)
        if self.enc_trace is None or self.enc_trace[0] != key:
            self.enc_trace = (key, encode_l2.simulate(N, R, kind, loops, self.l2_bytes(), 32, hints))
        trace = self.enc_trace[1]
        ckey = (N, R, kind, self.l2_bytes())
        if self.enc_comparison is None or self.enc_comparison[0] != ckey:
            rows = []
            for lp in (1 << i for i in range(13)):
                if lp > N >> R or lp > 256:
                    break
                rows.append((lp, encode_l2.simulate(N, R, kind, lp, self.l2_bytes(), 32, False),
                             encode_l2.simulate(N, R, kind, lp, self.l2_bytes(), 32, True)))
            self.enc_comparison = (ckey, rows)
        n = len(trace.iterations)
        self.enc_step.blockSignals(True)
        self.enc_step.setRange(0, n - 1)
        self.enc_step.blockSignals(False)
        it = self.enc_step.value()
        loop, level = trace.iterations[it]
        self.enc_step_label.setText(f"{it + 1} / {n}: loop {loop}, level {level}")
        self.enc_step_label.setMinimumWidth(170)
        self.encode_view.show_l2(trace, it)
        self.enc_info.setHtml(l2_html(trace, it, self.enc_comparison[1]))
        self.statusBar().showMessage("Move the iteration slider; change Loops to see the L2 reuse appear or vanish")

    def _compute_schedule(self, N, R, op):
        loops = min(self.enc_loops.currentData(), N >> R)
        G = grid.grid_blocks(self.sim_sms.value(), self.enc_blocks.value())
        kind = "decode" if op == "decode C" else "encode"
        pipelined = self.enc_pipelined.isChecked()
        steps = grid.schedule(N, R, kind, loops, G, pipelined)
        other = grid.schedule(N, R, kind, loops, G, not pipelined)
        self.enc_step.blockSignals(True)
        self.enc_step.setRange(0, len(steps) - 1)
        self.enc_step.blockSignals(False)
        current = self.enc_step.value()
        self.enc_step_label.setText(f"step {current + 1} / {len(steps)}")
        self.encode_view.show_schedule(N, R, op, loops, steps, current)
        sim_loops = tuple(min(b.currentData(), N >> R) for b in self.sim_loops)
        self.enc_info.setHtml(schedule_html(N, R, op, loops, steps, other, current, pipelined)
                              + passes_html(N, R, sim_loops, G))
        self.statusBar().showMessage("Move the step slider to run the schedule; hover for details")

    def compute_tiles(self):
        """One leaf product C = A * B (M = N >> R, ld = N) of the batched leaf GEMM"""
        N, R = self.n(), self.r()
        M = N >> R
        cfg = tile_order.CONFIGS[self.tile_config.currentIndex()]
        if M % cfg.BM or M % cfg.BN or M % cfg.BK:
            self.tile_view.set_message(f"{cfg.name} does not divide the leaf {M} x {M}.")
            self.tile_info.setText("")
            return
        order = self.tile_orders.currentData()
        image, s = tile_order.tile_order_image(M, cfg, order, self.tile_sms.value(), N)
        self.tile_view.set_image(image)
        self.tile_info.setText(f"{7 ** R} products of {M} x {M}; first wave: {s.a_panels} A + {s.b_panels} B panels, "
                               f"{s.slice_kb:.0f} KB per k slice, A span {s.a_span_mb:.1f} MB")
        self.statusBar().showMessage(f"Tile order: leaf {M}, {cfg.name}, {tile_order.order_name(order)}")

    # ---- files
    def save(self):
        tab = self.tabs.currentWidget()
        if tab is self.graph_tab:
            return self.save_graph()
        if tab is self.layout_tab:
            path, _ = QFileDialog.getSaveFileName(self, "Save layout", f"memory_layout_N{self.n()}_R{self.r()}.svg",
                                                  "SVG (*.svg);;PNG (*.png)")
            if path and not self.layout_view.save(path):
                QMessageBox.warning(self, "Save layout", f"Could not write {path}")
            return
        if tab is self.encode_tab:
            name = f"{self.enc_op.currentText().replace(' ', '_')}_N{self.n()}_R{self.r()}_level{self.enc_level.value()}"
            path, _ = QFileDialog.getSaveFileName(self, "Save image", name + ".png", "PNG (*.png)")
            if path and not self.encode_view.grab().save(path):
                QMessageBox.warning(self, "Save image", f"Could not write {path}")
            return
        if tab is self.sim_tab:
            name = f"l2_simulation_N{self.n()}_R{self.r()}_{self.l2.value():g}MB_{self.sim_leaf.currentData()}.html"
            path, _ = QFileDialog.getSaveFileName(self, "Save simulation", name, "HTML (*.html)")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.sim_view.toHtml())
            return
        view = self.tile_view
        if tab is not self.tile_tab or view.image is None:
            return
        name = (f"tile_order_N{self.n()}_R{self.r()}_config{self.tile_config.currentIndex()}"
                f"_order{self.tile_orders.currentData()}")
        path, _ = QFileDialog.getSaveFileName(self, "Save image", name + ".png", "Images (*.png *.bmp)")
        if path and not cv2.imwrite(path, view.image):
            QMessageBox.warning(self, "Save image", f"Could not write {path}")

    def save_graph(self):
        if not self.dot_text:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", f"deps_R{self.r()}.svg",
                                              "SVG (*.svg);;DOT (*.dot)")
        if not path:
            return
        if path.endswith(".dot") or not self.svg:
            with open(os.path.splitext(path)[0] + ".dot", "w") as f:
                f.write(self.dot_text)
        else:
            with open(path, "wb") as f:
                f.write(self.svg)


def sim_html(N, R, l2, bw, leaf, config, order, loops, stages, orders) -> str:
    """Tables of the L2 simulation tab"""
    def mb(v):
        return f"{v / MB:,.0f}"

    rows = ""
    for s in stages:
        ratio = s.total / s.minimum if s.minimum else 1.0
        color = "#c00" if ratio > 1.05 else "#000"
        rows += (f"<tr><td>{s.name}</td><td align='right'>{mb(s.reads)}</td><td align='right'>{mb(s.writes)}</td>"
                 f"<td align='right'><b>{mb(s.total)}</b></td><td align='right'>{mb(s.minimum)}</td>"
                 f"<td align='right' style='color:{color}'>{ratio:.2f}</td><td align='right'>{s.ms(bw):.2f}</td>"
                 f"<td align='right'>{mb(s.requested)}</td><td style='color:#777'>{s.detail}</td></tr>")
    total = sum(s.total for s in stages)
    minimum = sum(s.minimum for s in stages)
    rows += (f"<tr><td><b>total</b></td><td align='right'>{mb(sum(s.reads for s in stages))}</td>"
             f"<td align='right'>{mb(sum(s.writes for s in stages))}</td><td align='right'><b>{mb(total)}</b></td>"
             f"<td align='right'>{mb(minimum)}</td><td align='right'>{total / minimum:.2f}</td>"
             f"<td align='right'><b>{sum(s.ms(bw) for s in stages):.2f}</b></td>"
             f"<td align='right'>{mb(sum(s.requested for s in stages))}</td><td></td></tr>")
    best = min(s.total for _, s in orders)
    order_rows = ""
    for o, s in orders:
        name = tile_order.order_name(o)
        name = f"<b>{name}</b>" if o == order else name
        mark = " &#9733;" if s.total == best else ""
        order_rows += (f"<tr><td>{name}{mark}</td><td align='right'>{mb(s.total)}</td>"
                       f"<td align='right'>{s.total / s.minimum:.2f}</td><td align='right'>{s.ms(bw):.2f}</td></tr>")
    gemm = "mma leaf GEMM" if leaf == "mma" else "fused kernel"
    return (f"<h3>DRAM traffic, N={N}, R={R}, L2 {l2 / MB:g} MB, {gemm} config #{config}, "
            f"{tile_order.order_name(order)}</h3>"
            f"<p>Loops: encode {loops[0]}, bilinear {loops[1]}, reconstruct {loops[2]}, decode {loops[3]}. "
            "Sizes in MB; <i>x min</i>: DRAM traffic / minimum (every input read once, every result written once); "
            f"ms: DRAM traffic at {bw} GB/s, a lower bound (the GEMM stages are usually compute bound).</p>"
            "<table border='1' cellspacing='0' cellpadding='4'><tr><th>stage</th><th>read</th><th>write</th>"
            "<th>DRAM</th><th>minimum</th><th>x min</th><th>ms</th><th>requested</th><th>simulation</th></tr>"
            + rows + "</table>"
            f"<h3>Tile orders of the {gemm} (config #{config})</h3>"
            "<table border='1' cellspacing='0' cellpadding='4'><tr><th>order</th><th>DRAM</th><th>x min</th>"
            "<th>ms</th></tr>" + order_rows + "</table>"
            "<p style='color:#777'>Model (gemm_l2/l2_sim.py): fully associative LRU of the L2 size, write-back; "
            "passes simulated one row loop at a time (loops share no rows); GEMM blocks in launch order, one wave "
            "(SMs x blocks per SM) in step, per k chunk the wave's A and B panel chunks; every stage starts with an "
            "empty L2. Cache hints and the set-associative L2 are not modelled. <i>requested</i>: bytes the kernels "
            "load and store, hits included (GEMM stages: each panel chunk once per wave).</p>")


def main():
    if sys.platform == "win32":  # own taskbar entry and icon instead of python.exe's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("gemm.viewer")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON))
    viewer = Viewer()
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
