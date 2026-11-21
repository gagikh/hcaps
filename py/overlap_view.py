"""The "Early products" tab of viewer.py: which leaf products can run while the sums are formed
(gemm_l2/overlap.py), shown together with the data they read, and a model timeline of the serial
(BFS) and overlapped (GraphCAPS) schedules.

Everything is drawn in leaf x leaf blocks (leaf = N >> R), one scene unit per block:
  - A and B: 2^R x 2^R blocks each, available right after the encoder (stage 0);
  - the sums S1..S6 of every node, level by level: a node of level l is a 2 x 3 tile of sums
    (S1 S2 / S3 S4 / S5 S6), each sum 2^(R-l-1) blocks wide; they become available after
    sums level l + 1 (stage l + 1). This is the logical view; the Memory layout tab shows where
    the S areas are packed in device memory;
  - the 7^R leaf products, one row m1..m7 per level R-1 node, coloured by the stage they can start
    at = the latest stage of their two operands.
Click a product to frame its two operand blocks, or click a block to frame the product that reads it.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from gemm_l2.layout import layout, sums_blocks
from gemm_l2.overlap import OPERANDS, Timeline, expected_counts, products
from gemm_l2.reuse import analyse

# stage 0 = right after the encoder, then sums level 1, 2, ...
STAGE_COLORS = ("#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#8c564b")
SUMS_ALL = "#7f7f7f"  # the serial BFS sums pass over all levels


def stage_color(s):
    return QColor(STAGE_COLORS[s % len(STAGE_COLORS)])


def stage_text(s):
    return "after the encoder" if s == 0 else f"after sums level {s}"


class Clickable(QGraphicsRectItem):
    def __init__(self, rect, key, view):
        super().__init__(rect)
        self.key, self.view = key, view
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def mousePressEvent(self, event):
        self.view.select(self.key)
        event.accept()


class OverlapView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#fafafa"))
        self.on_select = None        # callback(html) for the info line
        self.items = {}              # key -> scene item; keys: ("P", index), ("A"/"B"/"S", cx, cy)
        self.links = {}              # product index -> [block keys], block key -> product index
        self.names = {}              # key -> description
        self.frames = []

    # ---- helpers (scene unit: one leaf block)
    def _box(self, x, y, w, h, color, tip="", key=None, pen_width=0.06):
        item = Clickable(QRectF(x, y, w, h), key, self) if key is not None else QGraphicsRectItem(QRectF(x, y, w, h))
        item.setBrush(QColor(color))
        pen = QPen(QColor("#ffffff"), pen_width)
        item.setPen(pen)
        if tip:
            item.setToolTip(tip)
        if key is None:
            item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(item)
        if key is not None:
            self.items[key] = item
        return item

    def _frame(self, x, y, w, h, color="#555555", width=0.12, dash=False):
        item = QGraphicsRectItem(QRectF(x, y, w, h))
        pen = QPen(QColor(color), width, Qt.DashLine if dash else Qt.SolidLine)
        item.setPen(pen)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(item)
        return item

    def _text(self, text, x, y, size, color="#000", bold=False, max_w=None):
        item = QGraphicsSimpleTextItem(text)
        item.setFont(QFont("Sans", 10, QFont.Bold if bold else QFont.Normal))
        item.setBrush(QColor(color))
        b = item.boundingRect()
        scale = size / b.height()
        if max_w:
            scale = min(scale, max_w / b.width())
        item.setScale(scale)
        item.setPos(x, y)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(item)
        return item

    def _centered(self, text, x, y, w, h, color="#fff", fraction=0.7):
        item = QGraphicsSimpleTextItem(text)
        item.setFont(QFont("Sans", 10))
        item.setBrush(QColor(color))
        b = item.boundingRect()
        scale = min(w * fraction / b.width(), h * fraction / b.height())
        item.setScale(scale)
        item.setPos(x + (w - b.width() * scale) / 2, y + (h - b.height() * scale) / 2)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(2)
        self.scene().addItem(item)

    # ---- drawing
    def show(self, N: int, R: int, tl: Timeline, label_paths: bool):
        scene = self.scene()
        scene.clear()
        self.items, self.links, self.names, self.frames = {}, {}, {}, []
        lay = layout(N, R)
        res = analyse(lay, 1)
        prods = products(R)
        blocks = 1 << R                 # A, B: blocks per side
        counts = expected_counts(R)
        unit = max(1.0, blocks / 8)     # text size, grows with the drawing
        width = max(4 * blocks + 6, 40.0)

        # product -> its operand blocks, block -> its product
        reader_of = {}
        for key, c in res.cells.items():
            if c.readers:
                reader_of[key] = c.reader
        index = {p.name: p.index for p in prods}

        # ---- A and B
        y = 0.0
        self._text("A, B (layer 0: ready after the encoder)", 0, y, 1.4 * unit, bold=True, max_w=width)
        y += 2.2 * unit
        for k, buf in enumerate("AB"):
            x0 = k * (blocks + 2 * unit)
            self._text(buf, x0, y, 1.2 * unit, bold=True)
            for cy in range(blocks):
                for cx in range(blocks):
                    key = (buf, cx, cy)
                    reader = reader_of.get(key, "")
                    desc = f"{buf} block ({cy}, {cx})"
                    self.names[key] = desc
                    self._box(x0 + cx, y + 1.6 * unit + cy, 1, 1, stage_color(0),
                              f"{desc}: available {stage_text(0)}; operand of {reader}", key)
                    self._centered("0", x0 + cx, y + 1.6 * unit + cy, 1, 1)
                    if reader:
                        self.links[key] = index[reader]
            self._frame(x0, y + 1.6 * unit, blocks, blocks)
        y += 1.6 * unit + blocks + 3 * unit

        # ---- sums, level by level (logical view)
        for level in range(R):
            q = 1 << (R - level - 1)        # blocks per side of one sum
            tile_w, tile_h = 2 * q, 3 * q
            gap = max(0.5, q / 4)
            per_row = max(1, int((width + gap) // (tile_w + gap)))
            nodes = lay.level_nodes(level)
            self._text(f"Layer {level + 1}: sums S1..S6 of the {len(nodes)} level {level} node(s)", 0, y,
                       1.4 * unit, bold=True, max_w=width)
            y += 2.2 * unit
            for node in nodes:
                tx = (node.index % per_row) * (tile_w + gap)
                ty = y + (node.index // per_row) * (tile_h + gap)
                for s, (name, r) in enumerate(sums_blocks(node)):
                    sx, sy = tx + (s % 2) * q, ty + (s // 2) * q
                    for by in range(q):
                        for bx in range(q):
                            key = ("S", r.x // lay.leaf + bx, r.y // lay.leaf + by)
                            reader = reader_of.get(key, "")
                            desc = f"{name} of node L{level}.{node.index}" + (f", block ({by}, {bx})" if q > 1 else "")
                            self.names[key] = desc
                            self._box(sx + bx, sy + by, 1, 1, stage_color(level + 1),
                                      f"{desc}: available {stage_text(level + 1)}; operand of {reader}", key)
                            self._centered(str(level + 1), sx + bx, sy + by, 1, 1)
                            if reader:
                                self.links[key] = index[reader]
                self._frame(tx, ty, tile_w, tile_h, "#333333", 0.08)
                if q > 1:
                    self._text(f"L{level}.{node.index}", tx, ty + tile_h + 0.1, 0.8 * unit, "#555")
            rows = math.ceil(len(nodes) / per_row)
            y += rows * (tile_h + gap) + 3 * unit
        data_h = y

        # ---- products, right of the data
        x0 = width + 8 * unit
        cols = max(1, round(math.sqrt(len(prods) // 7 / 7)))
        per_col = math.ceil(len(prods) // 7 / cols)
        cell = max(1.0, data_h * 0.5 / per_col) if per_col else 1.0
        cell = min(cell, 2.0 * unit)
        self._text(f"Leaf products ({len(prods)}): number = layer whose sums they need (0 = only A, B)", x0, 0,
                   1.4 * unit, bold=True,
                   max_w=max(cols * 9 * cell, 30 * unit))
        py0 = 2.2 * unit
        for p in prods:
            key = ("P", p.index)
            x = x0 + (p.node // per_col) * 9 * cell + p.child * cell
            yy = py0 + (p.node % per_col) * cell
            desc = f"{p.name} = {OPERANDS[p.child]} of node L{R - 1}.{p.node} (path {p.path_text()})"
            self.names[key] = desc
            self._box(x, yy, cell, cell, stage_color(p.stage), f"{desc}\ncan start {stage_text(p.stage)}", key)
            self._centered(".".join(str(c + 1) for c in p.path) if label_paths else str(p.stage), x, yy, cell, cell)
        self.links.update({("P", i): [] for i in range(len(prods))})
        for key, value in list(self.links.items()):
            if key[0] != "P":
                self.links[("P", value)].append(key)
        ly = py0 + per_col * cell + 2 * unit
        for s, c in enumerate(counts):
            self._box(x0, ly + s * 1.8 * unit, unit, unit, stage_color(s))
            self._text(f"{s}: {c} products", x0 + 1.6 * unit, ly + s * 1.8 * unit - 0.1 * unit, unit)
        ly += len(counts) * 1.8 * unit + 2 * unit

        # ---- timelines, under the products
        lw = max(cols * 9 * cell, 50 * unit)
        total = max(tl.serial_ms, tl.overlapped_ms)
        scale = lw / total
        lane_h = 2.4 * unit
        yt = ly
        for title, bars, end in (("Serial (BFS): all sums, then all leaf GEMMs", tl.serial, tl.serial_ms),
                                 ("Overlapped (GraphCAPS): group s after sums level s", tl.overlapped,
                                  tl.overlapped_ms)):
            self._text(f"{title}: {end:.2f} ms", x0, yt, 1.2 * unit, bold=True, max_w=lw)
            yt += 2 * unit
            for lane, name in (("sums", "sums"), ("gemm", "GEMMs")):
                self._text(name, x0 - 6 * unit, yt + 0.6 * unit, unit, max_w=5.5 * unit)
                self._box(x0, yt, lw, lane_h, "#eeeeee")
                for b in (b for b in bars if b.lane == lane):
                    serial_sums = lane == "sums" and title.startswith("Serial")
                    color = SUMS_ALL if serial_sums else stage_color(b.stage)
                    what = ("sums of all levels (one pass, L2 reuse)" if serial_sums
                            else f"sums level {b.stage}" if lane == "sums" else
                            f"leaf GEMM group {b.stage} ({counts[b.stage]} products)")
                    w = max((b.end - b.start) * scale, 0.05)
                    self._box(x0 + b.start * scale, yt, w, lane_h, color,
                              f"{what}: {b.start:.2f} .. {b.end:.2f} ms ({b.end - b.start:.2f} ms)", pen_width=0.02)
                    if w > 2.5 * unit:
                        self._centered(f"{'S' if lane == 'sums' else 'G'}{'' if serial_sums else b.stage}",
                                       x0 + b.start * scale, yt, w, lane_h)
                yt += lane_h + 0.5 * unit
            yt += 1.5 * unit
        step = 10 ** math.floor(math.log10(max(total / 5, 1e-6)))
        for k in range(int(total / step) + 1):
            t = k * step
            self._box(x0 + t * scale, yt - 1.6 * unit, 0.1 * unit, 0.6 * unit, "#555555", pen_width=0)
            self._text(f"{t:g}", x0 + t * scale - 0.3 * unit, yt - 0.9 * unit, 0.7 * unit, "#555")
        self._text("ms", x0 + lw + unit, yt - 0.9 * unit, 0.7 * unit, "#555")

        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-3 * unit, -3 * unit, 3 * unit, 3 * unit))
        self.fit()

    # ---- selection
    def select(self, key):
        for f in self.frames:
            self.scene().removeItem(f)
        self.frames = []
        product = key if key[0] == "P" else ("P", self.links.get(key))
        if product[1] is None:
            return
        operands = self.links.get(product, [])
        for k, color in [(product, "#000000")] + [(k, "#e6007e") for k in operands]:
            item = self.items.get(k)
            if item is None:
                continue
            r = item.rect()
            frame = self._frame(r.x(), r.y(), r.width(), r.height(), color, max(r.width() * 0.15, 0.15))
            frame.setZValue(10)
            self.frames.append(frame)
        if self.on_select:
            ops = [self.names[k] for k in operands]
            self.on_select(f"<b>{self.names[product]}</b>: reads " + " and ".join(ops) + " (pink frames)")

    def fit(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


def overlap_html(N: int, R: int, tl: Timeline) -> str:
    counts = expected_counts(R)
    rows = ""
    for s, c in enumerate(counts):
        sums = f"{tl.sums_ms[s]:.2f}" if s else "&ndash;"
        rows += (f"<tr><td><span style='color:{STAGE_COLORS[s % len(STAGE_COLORS)]}'>&#9632;</span> {s}</td>"
                 f"<td align='right'>{c}</td><td align='right'>{sums}</td>"
                 f"<td align='right'>{tl.gemm_ms[s]:.2f}</td></tr>")
    saved = tl.serial_ms - tl.overlapped_ms
    return (f"<b>Early leaf products, N={N}, R={R}</b><table cellspacing='4'>"
            "<tr><th align='left'>stage</th><th>products</th><th>sums ms</th><th>GEMM ms</th></tr>" + rows +
            f"</table>serial (BFS): <b>{tl.serial_ms:.2f} ms</b><br>overlapped (GraphCAPS): "
            f"<b>{tl.overlapped_ms:.2f} ms</b> ({'saves' if saved >= 0 else 'costs'} {abs(saved):.2f} ms)<br><br>"
            "<span style='color:#777'>Colours = when data exists: A and B after the encoder (green), the sums of "
            "level l after sums level l + 1. A product can start when both its operands exist. Click a product "
            "to see its operands, or a block to see its product. Model: GEMM at the given TFLOPS; sums at the "
            "given bandwidth with the minimal DRAM traffic, BFS re-reads earlier levels from L2, GraphCAPS (one "
            "kernel per level) from DRAM; the two streams are assumed not to slow each other down (optimistic). "
            "Encoder not included.</span><br><br>")
