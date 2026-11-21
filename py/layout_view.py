"""The "Memory layout" tab of viewer.py: the BFS device buffers (gemm_l2/layout.py) drawn to scale.

One scene unit is one half horizontally and one row vertically, so areas are bytes / 2. The S buffer
is one address range (row after row); it is folded into columns like text so it fits the view.
Click a node to highlight its parent and children; wheel zooms, drag pans.
"""

from __future__ import annotations

import math
import re

from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QTransform
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView)

from gemm_l2.layout import HALF, Layout, Rect, sums_blocks
from gemm_l2.reuse import LIVE, RECON, TAIL, Reuse, summary

MB = 1024 * 1024
LEVEL_COLORS = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948", "#76b7b2")
INPUT_COLOR, PRODUCT_COLOR, FREE_COLOR, L2_COLOR = "#9c9c9c", "#e15759", "#dddddd", "#e6007e"
# reuse overlay (gemm_l2/reuse.py): leaf blocks of A, B, S that can take products, those that cannot
REUSE_FREE, REUSE_BLOCKED, REUSE_RECON = QColor(44, 160, 44, 150), QColor(40, 40, 40, 110), QColor(140, 0, 0, 150)
# leaf operand overlay: left / right operand of a leaf multiplication, read by none
# opaque, so the level colours and the S1..S6 / node names underneath do not mix in
OPERAND_L, OPERAND_R, UNUSED_OPERAND = QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#d9d9d9")


def mb(n, html=False):
    text = f"{n / MB:,.1f} MB" if n >= MB else f"{n / 1024:,.1f} KB"
    return text.replace(" ", "&nbsp;") if html else text


def node_name(node):
    return f"L{node.level}.{node.index}"


class NodeItem(QGraphicsRectItem):
    """One piece of a node's ROI (a ROI split by a column fold has several)"""

    def __init__(self, rect, node, view):
        super().__init__(rect)
        self.node, self.view = node, view
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def mousePressEvent(self, event):
        self.view.select(self.node)
        event.accept()


class LayoutView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#fafafa"))
        self.lay = None
        self.zoom = 1.0          # relative to the fitted view
        self.fit_width = False   # fit the scene width instead of the whole scene
        self.on_zoom = None      # callback(zoom) when the wheel or Fit changes the zoom
        self.items = {}          # (level, index) -> [NodeItem]
        self.highlighted = []
        self.on_select = None    # callback(text)

    # ---- geometry
    def _fold(self, r: Rect):
        """Pieces (x, y, w, h) of an S rect in the folded scene"""
        pieces = []
        y, end = r.y, r.y + r.h
        while y < end:
            c = y // self.column_rows
            top = c * self.column_rows
            h = min(end, top + self.column_rows) - y
            pieces.append((self.s_left + c * self.column_step + r.x, y - top, r.w, h))
            y += h
        return pieces

    def _rect(self, x, y, w, h, color, pen=None, tip=None, brush_style=Qt.SolidPattern):
        item = QGraphicsRectItem(QRectF(x, y, w, h))
        item.setBrush(QBrush(QColor(color), brush_style))
        item.setPen(pen or QPen(Qt.NoPen))
        if tip:
            item.setToolTip(tip)
        self.scene().addItem(item)
        return item

    def _label(self, text, x, y, w, h, color="#000", fraction=0.6):
        """Text scaled to fit fraction of the rect (in scene units)"""
        item = QGraphicsSimpleTextItem(text)
        item.setFont(QFont("Sans", 10))
        item.setBrush(QColor(color))
        b = item.boundingRect()
        scale = min(w * fraction / b.width(), h * fraction / b.height())
        item.setScale(scale)
        item.setPos(x + (w - b.width() * scale) / 2, y + (h - b.height() * scale) / 2)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(item)
        return item

    def _heading(self, text, x, y, size, width):
        """Bold text above (x, y), lines size scene units high, wrapped at ", " to stay within
        width (no overlap with the next section's heading); shrunk only if one piece is wider"""
        font = QFont("Sans", 10, QFont.Bold)

        def measure(s):
            probe = QGraphicsSimpleTextItem(s)
            probe.setFont(font)
            b = probe.boundingRect()
            return b.width() * size / b.height()

        # break after ", " (dropped) or before " = " (kept as "= ")
        tokens = re.split(r"(, | = )", text)
        lines = [tokens[0]]
        for sep, part in zip(tokens[1::2], tokens[2::2]):
            if measure(lines[-1] + sep + part) <= width:
                lines[-1] += sep + part
            else:
                lines.append(part if sep == ", " else "= " + part)
        item = QGraphicsSimpleTextItem("\n".join(lines))
        item.setFont(font)
        b = item.boundingRect()
        scale = min(size * len(lines) / b.height(), width / b.width())
        item.setScale(scale)
        item.setPos(x, y - b.height() * scale - size * 0.3)
        self.scene().addItem(item)

    # ---- drawing
    def show_layout(self, lay: Layout, l2_bytes: int, columns: int = 0, show_inputs=True, show_sums=True,
                    reuse: Reuse = None, operands_only=False):
        """reuse: draw over S which leaf x leaf cells can hold leaf products (and, after
        gemm_l2.reuse.place(), the products placed there); None: plain layout"""
        scene = self.scene()
        scene.clear()
        self.items, self.highlighted, self.lay = {}, [], lay
        N = lay.N
        gap = N // 4
        heading = N / 10
        thin = QPen(QColor("#ffffff"))
        thin.setCosmetic(True)
        outline = QPen(QColor("#333333"))
        outline.setCosmetic(True)

        x = 0
        if show_inputs:  # A and B, N x N halves each
            for i, buf in enumerate((lay.A, lay.B)):
                y = i * (N + gap)
                self._rect(x, y, N, N, INPUT_COLOR, outline, f"{buf.name}: {N} x {N} halves, {mb(buf.bytes)}")
                self._label(f"{buf.name}\n{mb(buf.bytes)}", x, y, N, N, "#fff", 0.4)
            x += N + 2 * gap

        # S, folded into columns
        rows = lay.S.rows
        if columns <= 0:  # about as wide as the view is
            aspect = max(0.5, self.viewport().width() / max(1, self.viewport().height()))
            columns = max(1, round(math.sqrt(rows * aspect / N) * 0.8))
        self.column_rows = math.ceil(rows / columns)
        self.column_step = N + gap // 2
        self.s_left = x
        self._heading(f"S: {rows} rows x {N} halves = {mb(lay.S.bytes)}, folded into {columns} column(s)",
                      x, 0, heading, columns * self.column_step - gap // 2)
        for f in lay.free:  # plain fill: a pattern brush is in screen pixels and looks like blocks
            dashed = QPen(QColor("#999999"), 1, Qt.DashLine)
            dashed.setCosmetic(True)
            for p in self._fold(f):
                self._rect(*p, FREE_COLOR, dashed, f"unused: {f.h} rows x {f.w} halves, {mb(f.w * f.h * HALF)}")
                self._label(f"unused\n{mb(f.w * f.h * HALF)}", *p, "#777", 0.4)
        for node in lay.nodes:
            color = QColor(LEVEL_COLORS[node.level % len(LEVEL_COLORS)])
            tip = (f"{node_name(node)}: level {node.level} node {node.index}"
                   + (f", m{node.child + 1} = {node.operands} of L{node.level - 1}.{node.parent}"
                      if node.level else " (root: A * B)")
                   + f"\nS1..S6 of {node.rect.h // 3} x {node.rect.h // 3} halves, {mb(node.rect.w * node.rect.h * HALF)}"
                   + f"\nrows {node.rect.y}..{node.rect.y + node.rect.h - 1}, halves {node.rect.x}..{node.rect.x + node.rect.w - 1}"
                   + f"\nfirst byte at {mb(node.rect.y * lay.S.pitch + node.rect.x * HALF)}")
            pieces = []
            for p in self._fold(node.rect):
                item = NodeItem(QRectF(*p), node, self)
                item.setBrush(color)
                item.setPen(outline)
                item.setToolTip(tip)
                scene.addItem(item)
                pieces.append(item)
            self.items[(node.level, node.index)] = pieces
            # S1..S6 names where there are few nodes, else the node name
            sum_names = show_sums and (node.level <= 1 or len(lay.nodes) < 500)
            if show_sums:
                for name, r in sums_blocks(node):
                    for p in self._fold(r):
                        item = QGraphicsRectItem(QRectF(*p))
                        item.setPen(thin)
                        item.setAcceptedMouseButtons(Qt.NoButton)
                        scene.addItem(item)
                        if sum_names:
                            self._label(f"{name}\n{node_name(node)}", *p, "#fff", 0.45)
            if not sum_names:
                r = max(pieces, key=lambda i: i.rect().height()).rect()
                self._label(node_name(node), r.x(), r.y(), r.width(), r.height(), "#fff", 0.5)

        # the first L2 bytes of S
        # (to scale: it ends where the L2 size ends, not at a node boundary)
        l2_rows = min(rows, l2_bytes // lay.S.pitch)
        pen = QPen(QColor(L2_COLOR), 3, Qt.DashLine)
        pen.setCosmetic(True)
        for i, p in enumerate(self._fold(Rect(0, 0, N, l2_rows))):
            item = self._rect(*p, L2_COLOR, pen, f"L2 = {mb(l2_bytes)}: the first {l2_rows} rows of S",
                              Qt.NoBrush)
            item.setZValue(5)
            item.setAcceptedMouseButtons(Qt.NoButton)
            if i == 0:  # name it at its bottom right corner, inside S
                tag = QGraphicsSimpleTextItem(f"L2 {mb(l2_bytes)}")
                tag.setFont(QFont("Sans", 10, QFont.Bold))
                tag.setBrush(QColor(L2_COLOR))
                tb = tag.boundingRect()
                tag.setScale(heading * 0.6 / tb.height())
                tag.setPos(p[0] + p[2] - tb.width() * tag.scale() - gap / 8,
                           p[1] + min(p[3], l2_rows) - tb.height() * tag.scale() - gap / 16)
                tag.setZValue(6)
                tag.setAcceptedMouseButtons(Qt.NoButton)
                self.scene().addItem(tag)

        if reuse is not None and operands_only:
            self._draw_operands(reuse, lay, (0, N + gap) if show_inputs else None)
        elif reuse is not None:
            self._draw_reuse(reuse, lay, (0, N + gap) if show_inputs else None)

        # leaf products: one continuous buffer of 7 M x M per last-level node
        x = self.s_left + columns * self.column_step + gap
        M = lay.leaf
        per_column = max(1, self.column_rows // (7 * M + M // 2))
        p_columns = math.ceil(len(lay.products) / per_column)
        self._heading(f"P: {len(lay.products)} x 7 products of {M} x {M}, "
                      f"{mb(sum(p.bytes for p in lay.products))}", x, 0, heading,
                      max(p_columns * (M + M // 2) - M // 2, 4 * N))
        for i, buf in enumerate(lay.products):
            c, k = divmod(i, per_column)
            px, py = x + c * (M + M // 2), k * (7 * M + M // 2)
            parent = lay.level_nodes(lay.R - 1)[i]
            tip = f"{buf.name}: the 7 products of {node_name(parent)}, {mb(buf.bytes)} (own allocation)"
            for j in range(7):
                where = reuse.placed.get((i, j)) if reuse is not None and not operands_only else None
                if where is None:
                    self._rect(px, py + j * M, M, M, PRODUCT_COLOR, outline, tip)
                else:  # moved into S: only its outline stays here
                    ghost = QPen(QColor(PRODUCT_COLOR), 1, Qt.DashLine)
                    ghost.setCosmetic(True)
                    self._rect(px, py + j * M, M, M, "#ffffff", ghost,
                               f"P{i}.{j + 1}: stored in S at rows {where[1] * M}.., halves {where[0] * M}..",
                               Qt.NoBrush)
            if M >= N / 64:
                self._label(buf.name, px, py, M, 7 * M, "#fff", 0.8)

        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-gap, -gap, gap, gap))
        self._apply(keep_center=False)  # keep the zoom of the previous layout
        self.centerOn(scene.sceneRect().center().x(), scene.sceneRect().top())

    def _pieces(self, key, r, inputs_y):
        """Scene rects of a leaf block of A, B (unfolded, A at y = 0, B at inputs_y[1]) or S (folded)"""
        if key[0] == "S":
            return self._fold(r)
        return [(r.x, r.y + (inputs_y[1] if key[0] == "B" else 0), r.w, r.h)]

    def _draw_operands(self, res: Reuse, lay: Layout, inputs_y):
        """Every leaf x leaf block of A, B and S: blue = left operand, orange = right operand of a
        leaf multiplication (label: which one), grey hatched = read by none"""
        label_cells = len(res.cells) <= 6000
        border = QPen(QColor("#ffffff"), 1)
        border.setCosmetic(True)
        for key, c in res.cells.items():
            if key[0] != "S" and inputs_y is None:
                continue
            r = res.cell_rect(key)
            if c.readers:
                color = OPERAND_L if c.side == "L" else OPERAND_R
                tip = (f"{key[0]} block ({key[1]}, {key[2]}) {c.owner}: {'left' if c.side == 'L' else 'right'} "
                       f"operand of leaf multiplication {c.reader}" + (f" (+{c.readers - 1} more)" if c.readers > 1 else ""))
            else:
                color = QColor(UNUSED_OPERAND)
                tip = f"{key[0]} block ({key[1]}, {key[2]}) {c.owner or 'unused tail'}: not read by any leaf multiplication"
            for p in self._pieces(key, r, inputs_y):
                item = self._rect(*p, color, border, tip)
                item.setZValue(7)  # above the nodes, their S1..S6 frames and names, and the L2 frame
                item.setAcceptedMouseButtons(Qt.NoButton)
                if label_cells:
                    text = f"{c.reader}{c.side}" if c.readers else "unused"
                    self._label(text, *p, "#fff" if c.readers else "#777", 0.55).setZValue(8)

    def _draw_reuse(self, res: Reuse, lay: Layout, inputs_y):
        """One translucent square per leaf x leaf block of A, B and S: green = can take products
        (from which GEMM group on), dark = read by the last group, red = written by the leaf
        reconstruction; placed products drawn as red squares named P<node>.<i>.
        inputs_y: scene y of A and B (they are drawn at x = 0), None when they are hidden."""
        last = res.groups - 1
        label_cells = len(res.cells) <= 3000
        for key, c in res.cells.items():
            r = res.cell_rect(key)
            if key[0] != "S" and inputs_y is None:
                continue
            if c.status == TAIL:
                color, tip = REUSE_FREE, "unused tail of S: free for products from the start"
            elif c.status == RECON:
                color = REUSE_RECON
                tip = (f"{c.owner}: written by the leaf reconstruction while other products wait, "
                       "cannot hold products")
            elif c.last_group < last:
                color = QColor(REUSE_FREE)
                color.setAlpha(60 + int(90 * (1 - (c.last_group + 1) / max(1, res.groups))))
                tip = (f"{c.owner}: operand of {c.reader} only (GEMM group {c.last_group}); "
                       f"free for the products of groups {c.last_group + 1}..{last}")
            else:
                color = REUSE_BLOCKED
                tip = (f"{c.owner}: operand of {c.reader} only, in the last GEMM group {c.last_group}; "
                       "never free while products live")
            if c.product:
                color, tip = QColor(PRODUCT_COLOR), f"{c.product} stored here ({c.owner or 'unused tail'})"
            if key[0] == "S":
                pieces = self._fold(r)
            else:  # A at y = 0, B below it, unfolded
                pieces = [(r.x, r.y + (inputs_y[1] if key[0] == "B" else 0), r.w, r.h)]
            for p in pieces:
                item = self._rect(*p, color, None, tip)
                item.setZValue(3)
                item.setAcceptedMouseButtons(Qt.NoButton)
                if label_cells and c.product:
                    self._label(c.product, *p, "#fff", 0.6).setZValue(4)
                elif label_cells and c.status == LIVE and c.last_group < last and res.groups > 1:
                    self._label(f">{c.last_group}", *p, "#fff", 0.4).setZValue(4)

    # ---- zoom: 100% = the whole scene (or its width) fits the view
    def _base_scale(self):
        rect = self.scene().sceneRect()
        if rect.isEmpty():
            return 1.0
        sx = (self.viewport().width() - 2) / rect.width()
        sy = (self.viewport().height() - 2) / rect.height()
        return sx if self.fit_width else min(sx, sy)

    def _apply(self, keep_center=True):
        center = self.mapToScene(self.viewport().rect().center())
        scale = self._base_scale() * self.zoom
        self.setTransform(QTransform.fromScale(scale, scale))
        if keep_center:
            self.centerOn(center)

    def set_zoom(self, zoom, fit_width):
        changed_fit = fit_width != self.fit_width
        self.zoom, self.fit_width = zoom, fit_width
        self._apply(keep_center=not changed_fit)
        if changed_fit:
            self.centerOn(self.scene().sceneRect().center().x(), self.scene().sceneRect().top())

    def fit(self):
        self.zoom = 1.0
        self._apply(keep_center=False)
        self.centerOn(self.scene().sceneRect().center())
        if self.on_zoom:
            self.on_zoom(self.zoom)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    # ---- interaction
    def select(self, node):
        for item in self.highlighted:  # frames drawn over everything, so the labels stay visible
            self.scene().removeItem(item)
        self.highlighted = []
        lay = self.lay
        related = [(node, "#000000", 4)]
        if node.level:
            related.append((lay.level_nodes(node.level - 1)[node.parent], "#d62728", 3))
        related += [(n, "#00b7ff", 3) for n in lay.level_nodes(node.level + 1) if n.parent == node.index]
        for n, color, width in related:
            for piece in self.items.get((n.level, n.index), []):
                pen = QPen(QColor(color), width)
                pen.setCosmetic(True)
                frame = QGraphicsRectItem(piece.rect())
                frame.setPen(pen)
                frame.setZValue(10)
                frame.setAcceptedMouseButtons(Qt.NoButton)
                self.scene().addItem(frame)
                self.highlighted.append(frame)
        if self.on_select:
            children = [n for n in lay.level_nodes(node.level + 1) if n.parent == node.index]
            text = f"<b>{node_name(node)}</b> (black)"
            if node.level:
                text += f": m{node.child + 1} = {node.operands} of L{node.level - 1}.{node.parent} (red)"
            if children:
                spread = (max(c.rect.y for c in children) + children[0].rect.h - min(c.rect.y for c in children))
                text += (f"<br>children (blue) {children[0].index}..{children[-1].index} span "
                         f"{mb(spread * lay.S.pitch)} of S")
            elif node.level == lay.R - 1:
                text += f"<br>products in P{node.index}"
            self.on_select(text)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        factor = min(max(self.zoom * factor, 0.1), 50.0) / self.zoom
        self.zoom *= factor
        self.scale(factor, factor)  # around the mouse (AnchorUnderMouse)
        if self.on_zoom:
            self.on_zoom(self.zoom)

    def save(self, path):
        rect = self.scene().sceneRect()
        if path.lower().endswith(".svg"):
            generator = QSvgGenerator()
            generator.setFileName(path)
            width = 2000
            generator.setSize(QSize(width, int(width * rect.height() / rect.width())))
            generator.setViewBox(QRectF(0, 0, width, width * rect.height() / rect.width()))
            painter = QPainter(generator)
            self.scene().render(painter, QRectF(), rect)
            painter.end()
            return True
        width = 4000
        image = QImage(width, max(1, int(width * rect.height() / rect.width())), QImage.Format_ARGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.scene().render(painter, QRectF(image.rect()), rect)
        painter.end()
        return image.save(path)


def operands_html(res: Reuse) -> str:
    """Counts of the leaf operand test for the side panel"""
    lines = [f"<b>Leaf operands: blocks of {res.leaf} x {res.leaf} read by leaf multiplications</b><br>"]
    for buf in ("A", "B", "S"):
        cells = [c for k, c in res.cells.items() if k[0] == buf]
        used = sum(1 for c in cells if c.readers)
        multi = sum(1 for c in cells if c.readers > 1)
        lines.append(f"{buf}: {used} of {len(cells)} used" + (f", {multi} read twice or more" if multi else "")
                     + (f", unused {len(cells) - used} ({mb((len(cells) - used) * res.cell_bytes())})"
                        if used < len(cells) else "") + "<br>")
    reads = sum(c.readers for c in res.cells.values())
    lines.append(f"operand reads: {reads} = 2 x {res.products} products<br>"
                 f"<span style='color:#1f77b4'>&#9632;</span> left operand, "
                 f"<span style='color:#ff7f0e'>&#9632;</span> right operand, "
                 f"<span style='color:#777'>&#9632;</span> read by none (hatched); label P&lt;node&gt;.&lt;i&gt;L/R<br><br>")
    return "".join(lines)


def reuse_html(res: Reuse) -> str:
    """Summary of gemm_l2.reuse for the side panel"""
    s = summary(res)
    cell = res.cell_bytes()
    text = (f"<b>Leaf blocks ({res.leaf} x {res.leaf}) for products, {s['groups']} GEMM group(s)</b><br>"
            f"A {s['A_blocks']}, B {s['B_blocks']}, S {s['S_blocks']} blocks; read by no leaf multiplication: "
            f"<b>{s['never_read']}</b>, by more than one: {s['read_more_than_once']} "
            f"(every block of A, B and S is the operand of exactly one)<br>"
            f"<span style='color:#2ca02c'>&#9632;</span> unused tail of S: {s['tail']} ({mb(s['tail'] * cell)})<br>"
            f"<span style='color:#8c0000'>&#9632;</span> level {res.lay.R - 1} S1..S4 (leaf reconstruction "
            f"target): {s['recon']}<br>"
            f"&#9632; green: its reader is in an earlier group (label: that group), dark: in the last group<br>")
    if res.placement_done:
        text += (f"products placed: <b>{s['placed']} of {s['products']}</b> "
                 f"({100 * s['placed'] / s['products']:.0f}%: S {s['S_placed']}, A {s['A_placed']}, "
                 f"B {s['B_placed']}), saves {mb(s['saved_bytes'])} of {mb(s['products_bytes'])}<br>"
                 f"total {mb(s['total_before'])} &rarr; <b>{mb(s['total_after'])}</b><br>")
    text += ("<span style='color:#777'>One group = today's single batched GEMM: every block is still to be read "
             "until it ends, so only the tail of S is free. A product cannot overwrite its own operands (a tiled "
             "GEMM reads them for many output tiles). Needs per-product addresses in the GEMM and the leaf "
             "reconstruction; A and B lose the encoded inputs.</span><br><br>")
    return text


def summary_html(lay: Layout, l2_bytes: int) -> str:
    rows = ""
    for level in range(lay.R):
        nodes = lay.level_nodes(level)
        size = nodes[0].rect.w * nodes[0].rect.h * HALF
        rows += (f"<tr><td><span style='color:{LEVEL_COLORS[level % len(LEVEL_COLORS)]}'>&#9632;</span> "
                 f"{level}</td><td align='right'>{len(nodes)}</td><td align='right'>{size / MB:,.1f}</td>"
                 f"<td align='right'>{size * len(nodes) / MB:,.1f}</td></tr>")
    free = sum(f.w * f.h for f in lay.free) * HALF
    products = sum(p.bytes for p in lay.products)
    return (f"<b>Memory, N={lay.N}, R={lay.R}</b><table cellspacing='4'>"
            "<tr><th align='left'>S level</th><th>nodes</th><th>each, MB</th><th>total, MB</th></tr>" + rows +
            f"<tr><td>unused</td><td></td><td></td><td align='right'>{free / MB:,.1f}</td></tr></table>"
            f"A + B: {mb(lay.A.bytes + lay.B.bytes)}<br>S: {mb(lay.S.bytes)} ({lay.S.rows} rows, "
            f"{lay.S.bytes / l2_bytes:.1f} x L2)<br>P: {mb(products)} in {len(lay.products)} buffers<br>"
            f"<b>total {mb(lay.total_bytes())}</b><br><br>"
            "<span style='color:#777'>Dashed pink frame: the first L2-size bytes of S. Click a node: parent red, "
            "children blue.</span>")
