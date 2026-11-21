"""The "Encoder" tab of viewer.py: the alternative basis transforms (gemm_l2/encode.py) level by level.

The matrix is drawn as its C x C leaf blocks (C = 2^R). At level l every 2M x 2M group (thick frame)
is transformed: the blocks read with + or - and the ones written are coloured; blocks changed by an
earlier level are tinted. The rows of one row loop can be overlaid. Hover a block for what it holds
before and after the level in terms of the original blocks. Wheel zooms, drag pans.
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                             QGraphicsView)

from gemm_l2 import encode, grid

MB = 1024 * 1024
SIZE = 1000.0  # scene units per matrix side
ROLE_COLORS = {"": "#e4e4e4", "+": "#4e79a7", "-": "#e15759", "d": "#edc948", "w": "#edc948", "r": "#76b7b2"}
CHANGED_COLOR = "#c9e4c5"
ROLE_TEXT = {"": "not touched", "+": "read, added", "-": "read, subtracted", "d": "read and written (d)",
             "w": "read and written", "r": "read"}
PREFIX = {"encode A": "A", "encode B": "B", "decode C": "C"}
LEVEL_COLORS = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948", "#76b7b2")
MAX_LINES = 40000  # row lines of a loop drawn; above that the blocks are shaded


class EncodeView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#fafafa"))
        self.fitted = False

    def _labels(self, labels, fraction, line):
        """[(text, x, y, w, h, color)]: lines `line` scene units high, a label that does not fit
        (fraction of its cell) shrunk until it does"""
        items = []
        for text, x, y, w, h, color in labels:
            item = QGraphicsSimpleTextItem(text)
            item.setFont(QFont("Sans", 10))
            item.setBrush(QColor(color))
            b = item.boundingRect()
            fit = min(w * fraction / b.width(), h * fraction / b.height())
            items.append((item, b, x, y, w, h, fit))
        probe = QGraphicsSimpleTextItem("A")
        probe.setFont(QFont("Sans", 10))
        common = line / probe.boundingRect().height()
        for item, b, x, y, w, h, fit in items:
            scale = min(common, fit)
            item.setScale(scale)
            item.setPos(x + (w - b.width() * scale) / 2, y + (h - b.height() * scale) / 2)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.scene().addItem(item)

    def show_transform(self, N, R, operation, level, loops, loop, stripes, texts, states, its):
        """level 0: the input; level l: the roles of level l, contents after it. its: the
        grid.Iteration of every level of one loop (drawn below the matrix)"""
        scene = self.scene()
        scene.clear()
        C = 1 << R
        cell = SIZE / C
        prefix = PREFIX[operation]
        steps = encode.levels(R, operation)
        step = steps[level - 1] if level else None
        before, after = states[max(0, level - 1)], states[level]
        outline = QPen(QColor("#ffffff"))
        outline.setCosmetic(True)
        labels = []
        for (i, j), terms in after.items():
            role = step.roles[(i, j)] if step else ""
            changed = terms != states[0][(i, j)]
            color = ROLE_COLORS[role] if role else (CHANGED_COLOR if changed else ROLE_COLORS[""])
            item = QGraphicsRectItem(QRectF(j * cell, i * cell, cell, cell))
            item.setBrush(QBrush(QColor(color)))
            item.setPen(outline)
            name = encode.block_name(prefix, i, j, R)
            tip = f"{name}: {ROLE_TEXT[role] if step else 'input'}"
            if step and before[(i, j)] != terms:
                tip += (f"\nbefore level {level}: {encode.expression_text(before[(i, j)], prefix, R)}"
                        f"\nafter:  {encode.expression_text(terms, prefix, R)}")
            else:
                tip += f"\nholds: {encode.expression_text(terms, prefix, R)}"
            item.setToolTip(tip)
            scene.addItem(item)
            dark = role in ("+", "-")
            fg = "#ffffff" if dark else "#222222"
            if texts and C <= 4:
                expr = encode.expression_text(terms, prefix, R)
                if len(expr) > 20:  # one term per line
                    expr = expr.replace(" + ", "\n+ ").replace(" − ", "\n− ")
                text = f"{name}\n= {expr}" if expr != name else name
                labels.append((text, j * cell, i * cell, cell, cell, fg))
            elif C <= 16:
                labels.append((name, j * cell, i * cell, cell, cell, fg))
        self._labels(labels, 0.85, cell * 0.12)
        # the 2M x 2M groups of this level
        if step:
            frame = QPen(QColor("#222222"), 3)
            frame.setCosmetic(True)
            g = 2 * step.size * cell
            for y in range(0, C, 2 * step.size):
                for x in range(0, C, 2 * step.size):
                    item = QGraphicsRectItem(QRectF(x * cell, y * cell, g, g))
                    item.setPen(frame)
                    item.setAcceptedMouseButtons(Qt.NoButton)
                    item.setZValue(5)
                    scene.addItem(item)
        # rows of loop `loop`: rows loop, loop + loops, ... of the blocks this level touches
        if stripes and step:
            touched = [(i, j) for (i, j), r in step.roles.items() if r]
            rows_per_block = (N >> R) // loops  # rows of the loop inside one leaf block
            if len(touched) * rows_per_block <= MAX_LINES:
                path = QPainterPath()
                for i, j in touched:
                    for k in range(rows_per_block):
                        y = (i * (N >> R) + loop + k * loops + 0.5) * SIZE / N
                        path.moveTo(j * cell, y)
                        path.lineTo((j + 1) * cell, y)
                item = QGraphicsPathItem(path)
                pen = QPen(QColor(0, 0, 0, 70), 1)
                pen.setCosmetic(True)  # one pixel at every zoom: zoom in to see single rows
                item.setPen(pen)
            else:  # too many rows to draw: shade the blocks
                path = QPainterPath()
                for i, j in touched:
                    path.addRect(QRectF(j * cell, i * cell, cell, cell))
                item = QGraphicsPathItem(path)
                item.setBrush(QBrush(QColor(0, 0, 0, 50), Qt.Dense5Pattern))
                item.setPen(QPen(Qt.NoPen))
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setZValue(6)
            scene.addItem(item)
        title = QGraphicsSimpleTextItem(f"{operation}, " + (f"level {level} of {R}: {step.formula}" if step
                                                              else "input"))
        title.setFont(QFont("Sans", 10, QFont.Bold))
        title.setScale(SIZE / 40 / title.boundingRect().height())
        title.setPos(0, -SIZE / 40 * 1.6)
        scene.addItem(title)
        self._grid_strip(its, level, loops)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        if not self.fitted:
            self.fit()
            self.fitted = True

    def _text(self, text, x, y, height, bold=False, color="#222222"):
        item = QGraphicsSimpleTextItem(text)
        item.setFont(QFont("Sans", 10, QFont.Bold if bold else QFont.Normal))
        item.setBrush(QColor(color))
        item.setScale(height / item.boundingRect().height())
        item.setPos(x, y)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(item)
        return item

    def _grid_strip(self, its, level, loops):
        """Below the matrix: per level of one loop the grid-stride rounds of the G blocks, the busy
        part green, the blocks waiting at the barrier grey"""
        if not its:
            return
        G = its[0].grid
        row, gap, left = SIZE / 34, SIZE / 120, SIZE / 5
        y = SIZE + SIZE / 25
        s = grid.summary(its, loops)
        self._text(f"Grid: {G} blocks of 4 warps, per level of one loop (x {loops} loops): "
                   f"{s['utilization'] * 100:.0f}% of the block slots busy, {s['barriers']} grid barriers",
                   0, y, row * 0.8, True)
        y += row * 1.4
        busy_brush, idle_brush = QBrush(QColor("#59a14f")), QBrush(QColor("#c8c8c8"))
        pen = QPen(QColor("#ffffff"))
        pen.setCosmetic(True)
        width = SIZE - left
        for it in its:
            current = it.level == level
            self._text(f"level {it.level}", 0, y, row * 0.7, current)
            self._text(f"{it.work} blocks, {it.rounds} round{'s' if it.rounds > 1 else ''}", 0, y + row * 0.75,
                       row * 0.55, False, "#555555")
            self._text(f"{it.utilization * 100:.0f}% busy", 0, y + row * 1.35, row * 0.55, False, "#555555")
            rounds = list(range(it.rounds))
            if len(rounds) > 5:  # first two, the rest of the full ones elided, the last
                rounds = [0, 1, None, it.rounds - 1]
            top = y
            for r in rounds:
                if r is None:
                    self._text(f"\u22ee {it.rounds - 3} more full rounds", left + width / 2 - SIZE / 10, y,
                               row * 0.7, False, "#555555")
                    y += row * 0.9
                    continue
                busy = min(G, it.work - r * G)
                for x0, w, brush in ((0, width * busy / G, busy_brush), (width * busy / G, width * (G - busy) / G,
                                                                            idle_brush)):
                    if w <= 0:
                        continue
                    item = QGraphicsRectItem(QRectF(left + x0, y, w, row * 0.8))
                    item.setBrush(brush)
                    item.setPen(pen)
                    item.setToolTip(f"level {it.level}, round {r + 1} of {it.rounds}: {busy} of {G} blocks busy, "
                                    f"{G - busy} wait at the barrier")
                    self.scene().addItem(item)
                y += row
            if current:
                frame = QPen(QColor("#222222"), 2)
                frame.setCosmetic(True)
                item = QGraphicsRectItem(QRectF(left - 4, top - 4, width + 8, y - top + 4))
                item.setPen(frame)
                item.setAcceptedMouseButtons(Qt.NoButton)
                self.scene().addItem(item)
            y += gap * 3

    def show_schedule(self, N, R, operation, loops, steps, current):
        """How the kernel runs the transform: the matrix with the rows every active level touches in
        step `current` (one colour per level), the schedule of all steps (levels x steps, number =
        row loop) and the grid of the current step"""
        scene = self.scene()
        scene.clear()
        C = 1 << R
        cell = SIZE / C
        step = steps[current]
        levels = encode.levels(R, operation)
        G = step.grid
        # matrix: untouched blocks grey
        outline = QPen(QColor("#ffffff"))
        outline.setCosmetic(True)
        for i in range(C):
            for j in range(C):
                item = QGraphicsRectItem(QRectF(j * cell, i * cell, cell, cell))
                item.setBrush(QBrush(QColor("#ececec")))
                item.setPen(outline)
                item.setToolTip(encode.block_name(PREFIX[operation], i, j, R))
                scene.addItem(item)
        rows_per_leaf = N >> R
        for level, loop, work in step.parts:
            color = QColor(LEVEL_COLORS[(level - 1) % len(LEVEL_COLORS)])
            lv = levels[level - 1]
            fill = QColor(color)
            fill.setAlpha(70)
            written = [k for k, r in lv.roles.items() if r in ("d", "w")]
            read = [k for k, r in lv.roles.items() if r in ("+", "-", "r")]
            dashed = QPen(color, 2, Qt.DashLine)
            dashed.setCosmetic(True)
            for (i, j) in written:
                item = QGraphicsRectItem(QRectF(j * cell, i * cell, cell, cell))
                item.setBrush(QBrush(fill))
                item.setPen(QPen(Qt.NoPen))
                item.setToolTip(f"{encode.block_name(PREFIX[operation], i, j, R)}: written by level {level} "
                                f"(row loop {loop})")
                scene.addItem(item)
            for (i, j) in read:
                item = QGraphicsRectItem(QRectF(j * cell + 2, i * cell + 2, cell - 4, cell - 4))
                item.setPen(dashed)
                item.setToolTip(f"{encode.block_name(PREFIX[operation], i, j, R)}: read by level {level} "
                                f"(row loop {loop})")
                scene.addItem(item)
            frame = QPen(color.darker(130), 3)
            frame.setCosmetic(True)
            g = 2 * lv.size * cell
            for y0 in range(0, C, 2 * lv.size):
                for x0 in range(0, C, 2 * lv.size):
                    item = QGraphicsRectItem(QRectF(x0 * cell, y0 * cell, g, g))
                    item.setPen(frame)
                    item.setAcceptedMouseButtons(Qt.NoButton)
                    item.setZValue(5)
                    scene.addItem(item)
        # row ruler: the first rows of a leaf block, coloured by the level that handles them in this step
        # (row r belongs to row loop r mod loops)
        shown = min(2 * loops, 64, rows_per_leaf)
        rx, rw, rh = SIZE + SIZE / 14, SIZE / 9, SIZE / shown
        self._text("rows of a block", rx, -SIZE / 40 * 1.6, SIZE / 50, True)
        owner = {k: lv for lv, k, _ in step.parts}
        row_labels = []
        light = QPen(QColor("#ffffff"))
        light.setCosmetic(True)
        for r in range(shown):
            lv = owner.get(r % loops)
            color = QColor(LEVEL_COLORS[(lv - 1) % len(LEVEL_COLORS)]) if lv else QColor("#ececec")
            item = QGraphicsRectItem(QRectF(rx, r * rh, rw, rh))
            item.setBrush(QBrush(color))
            item.setPen(light)
            item.setToolTip(f"row {r}: row loop {r % loops}" + (f", level {lv} in this step" if lv else ", idle in this step"))
            scene.addItem(item)
            if rh >= SIZE / 70:
                row_labels.append((f"row {r}" + (f"  L{lv}" if lv else ""), rx, r * rh, rw, rh,
                                   "#ffffff" if lv else "#777777"))
        self._labels(row_labels, 0.8, rh * 0.45)
        if shown < rows_per_leaf:
            self._text(f"... {rows_per_leaf} rows per leaf block,", rx, SIZE + SIZE / 120, SIZE / 70, False, "#555555")
            self._text(f"repeating every {loops} rows", rx, SIZE + SIZE / 120 + SIZE / 55, SIZE / 70, False, "#555555")
        parts = ", ".join(f"level {lv} on row loop {k}" for lv, k, _ in step.parts)
        self._text(f"{operation}, step {current + 1} of {len(steps)}: {parts}", 0, -SIZE / 40 * 1.6, SIZE / 40, True)

        # schedule: levels x steps, the number in a cell = row loop
        row = SIZE / 34
        y = SIZE + SIZE / 25
        s = grid_summary(steps)
        self._text(f"Schedule: {len(steps)} steps (grid barriers), {s['utilization'] * 100:.0f}% of the block slots "
                   f"busy. Rows: levels, columns: steps, number: row loop", 0, y, row * 0.8, True)
        y += row * 1.4
        left = SIZE / 9
        width = (SIZE - left) / len(steps)
        numbers = width * 1.0 >= row * 0.9
        for lv in range(1, R + 1):
            self._text(f"level {lv}", 0, y + (lv - 1) * row, row * 0.65)
        for st in steps:
            for level, loop, work in st.parts:
                color = QColor(LEVEL_COLORS[(level - 1) % len(LEVEL_COLORS)])
                if st.index != current:
                    color.setAlpha(110)
                item = QGraphicsRectItem(QRectF(left + st.index * width, y + (level - 1) * row, width, row * 0.9))
                item.setBrush(QBrush(color))
                item.setPen(light if width > 3 else QPen(Qt.NoPen))
                item.setToolTip(f"step {st.index + 1}: level {level}, row loop {loop}, {work} blocks")
                scene.addItem(item)
                if numbers:
                    self._labels([(str(loop), left + st.index * width, y + (level - 1) * row, width, row * 0.9,
                                   "#ffffff")], 0.8, row * 0.5)
        frame = QPen(QColor("#222222"), 2)
        frame.setCosmetic(True)
        item = QGraphicsRectItem(QRectF(left + current * width - 1, y - 3, width + 2, R * row + 4))
        item.setPen(frame)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(7)
        scene.addItem(item)
        y += R * row + row * 0.8

        # the grid in the current step: rounds of G blocks, coloured by level, idle grey
        self._text(f"Grid in step {current + 1}: {step.work} virtual blocks for {G} blocks, {step.rounds} "
                   f"round{'s' if step.rounds > 1 else ''}, {step.utilization * 100:.0f}% busy", 0, y, row * 0.75, True)
        y += row * 1.3
        bar = SIZE - left
        segments = [(QColor(LEVEL_COLORS[(lv - 1) % len(LEVEL_COLORS)]), w, lv) for lv, _, w in step.parts]
        shown = list(range(step.rounds))
        if len(shown) > 5:
            shown = [0, 1, None, step.rounds - 1]
        for r in shown:
            if r is None:
                self._text(f"\u22ee {step.rounds - 3} more full rounds", left + bar / 2 - SIZE / 10, y, row * 0.7,
                           False, "#555555")
                y += row * 0.9
                continue
            self._text(f"round {r + 1}", 0, y, row * 0.6)
            lo, hi = r * G, min((r + 1) * G, step.work)  # virtual blocks of this round
            x, offset = left, 0
            for color, w, lv in segments:
                a, b = max(lo, offset), min(hi, offset + w)
                if b > a:
                    item = QGraphicsRectItem(QRectF(x, y, bar * (b - a) / G, row * 0.8))
                    item.setBrush(QBrush(color))
                    item.setPen(light)
                    item.setToolTip(f"round {r + 1}: {b - a} blocks of level {lv}")
                    scene.addItem(item)
                    x += bar * (b - a) / G
                offset += w
            if hi - lo < G:
                item = QGraphicsRectItem(QRectF(x, y, bar * (G - (hi - lo)) / G, row * 0.8))
                item.setBrush(QBrush(QColor("#c8c8c8")))
                item.setPen(light)
                item.setToolTip(f"round {r + 1}: {G - (hi - lo)} blocks wait at the barrier")
                scene.addItem(item)
            y += row
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        if not self.fitted:
            self.fit()
            self.fitted = True

    def show_l2(self, trace, it):
        """L2 of the original encoder / decoder after iteration it (loop, level): the matrices as
        cells (red = read from DRAM in this iteration, green = read from L2, blue = in L2, shade = the
        part of the cell; grey = not in L2), then the DRAM and L2 bytes of every iteration"""
        scene = self.scene()
        scene.clear()
        loop, level = trace.iterations[it]
        C = trace.cells
        nm = len(trace.matrices)
        gap = SIZE / 20
        size = (SIZE * 1.6 - gap * (nm - 1)) / nm if nm > 1 else SIZE
        cell = size / C
        cb = trace.cell_bytes
        mb = 1024 * 1024
        light = QPen(QColor("#ffffff"))
        light.setCosmetic(True)
        red, green, blue, grey = QColor("#e15759"), QColor("#59a14f"), QColor("#4e79a7"), QColor("#e6e6e6")

        def mix(a, b, f):
            return QColor(int(a.red() + (b.red() - a.red()) * f), int(a.green() + (b.green() - a.green()) * f),
                          int(a.blue() + (b.blue() - a.blue()) * f))

        h, mi, w = trace.hit[it], trace.miss[it], trace.written[it]
        res, drt = trace.resident[it], trace.dirty[it]
        for m, name in enumerate(trace.matrices):
            x0 = m * (size + gap)
            self._text(name, x0, -SIZE / 40 * 1.5, SIZE / 40, True)
            for band in range(C):
                for col in range(C):
                    read = h[m, band, col] + mi[m, band, col]
                    r = res[m, band, col] / cb
                    if read:
                        color = mix(red, green, h[m, band, col] / read)
                    elif w[m, band, col]:
                        color = QColor("#f28e2b")
                    elif r > 0:
                        color = mix(grey, blue, 0.25 + 0.75 * min(1.0, r * 8))
                    else:
                        color = grey
                    item = QGraphicsRectItem(QRectF(x0 + col * cell, band * cell, cell, cell))
                    item.setBrush(QBrush(color))
                    item.setPen(light)
                    item.setToolTip(f"{name} cell ({band}, {col}), {cb / mb:.2f} MB: read from L2 {h[m, band, col] / mb:.2f} MB, "
                                    f"from DRAM {mi[m, band, col] / mb:.2f} MB, written {w[m, band, col] / mb:.2f} MB; "
                                    f"in L2 after: {res[m, band, col] / mb:.2f} MB ({drt[m, band, col] / mb:.2f} dirty)")
                    scene.addItem(item)
                    if trace.last_use[it][m, band, col]:  # last access: evict-first hint
                        mark = QGraphicsRectItem(QRectF(x0 + col * cell + cell * 0.3, band * cell + cell * 0.3,
                                                        cell * 0.4, cell * 0.4))
                        mark.setBrush(QBrush(QColor("#7b3294")))
                        mark.setPen(QPen(Qt.NoPen))
                        mark.setAcceptedMouseButtons(Qt.NoButton)
                        mark.setZValue(4)
                        scene.addItem(mark)
            # the groups of this level
            frame = QPen(QColor("#222222"), 2)
            frame.setCosmetic(True)
            g = size / (1 << (level - 1))
            for gy in range(1 << (level - 1)):
                for gx in range(1 << (level - 1)):
                    item = QGraphicsRectItem(QRectF(x0 + gx * g, gy * g, g, g))
                    item.setPen(frame)
                    item.setAcceptedMouseButtons(Qt.NoButton)
                    item.setZValue(5)
                    scene.addItem(item)
        width = nm * size + (nm - 1) * gap
        reads = trace.hit_bytes[it] + trace.miss_bytes[it]
        rate = f"{100 * trace.hit_bytes[it] / reads:.0f}%" if reads else "-"
        self._text(f"{trace.operation}, row loop {loop} of {trace.loops}, level {level}: rows {loop}, {loop + trace.loops}, "
                   f"... of every group. Hit rate {rate} (all: {trace.hit_rate * 100:.0f}%). L2 "
                   f"{trace.l2_bytes / mb:g} MB holds {res.sum() / mb:,.0f} MB after it"
                   + (", last-use hints" if trace.evict_first else ""),
                   0, -SIZE / 40 * 3.2, SIZE / 42, True)
        y = size + SIZE / 30
        legend = [(red, "read from DRAM (miss)"), (green, "read from L2 (hit)"), (QColor("#f28e2b"), "written only"),
                  (mix(grey, blue, 0.8), "in L2"), (grey, "not in L2")]
        if trace.evict_first:
            legend.append((QColor("#7b3294"), "last access: __ldlu / __stcs"))
        x = 0
        for color, text in legend:
            item = QGraphicsRectItem(QRectF(x, y, SIZE / 50, SIZE / 50))
            item.setBrush(QBrush(color))
            item.setPen(QPen(Qt.NoPen))
            scene.addItem(item)
            label = self._text(text, x + SIZE / 40, y - SIZE / 400, SIZE / 50)
            x += SIZE / 40 + label.boundingRect().width() * label.scale() + SIZE / 30
        # bytes of every iteration: DRAM reads (red) and L2 hits (green) up, write backs (orange) down
        y += SIZE / 12
        n = len(trace.iterations)
        top = max(1, int((trace.hit_bytes + trace.miss_bytes).max()))
        bottom = max(1, int(trace.writeback_bytes.max()))
        hgt, low = SIZE / 4, SIZE / 12
        bw = width / n
        self._text(f"Every iteration (loop x level): read from DRAM (red), from L2 (green), written back (orange), "
                   f"hit rate (line). "
                   f"Total DRAM: read {trace.miss_bytes.sum() / mb:,.0f} MB, written {(trace.writeback_bytes.sum() + trace.flush_bytes) / mb:,.0f} MB, "
                   f"L2 hits {trace.hit_bytes.sum() / mb:,.0f} MB", 0, y - SIZE / 30, SIZE / 55, True)
        base = y + hgt
        for i, (lp, lv) in enumerate(trace.iterations):
            x = i * bw
            mh = hgt * trace.miss_bytes[i] / top
            hh = hgt * trace.hit_bytes[i] / top
            wh = low * trace.writeback_bytes[i] / bottom
            alpha = 255 if i == it else 150
            for y0, hh_, color in ((base - mh, mh, red), (base - mh - hh, hh, green), (base, wh, QColor("#f28e2b"))):
                if hh_ <= 0:
                    continue
                c = QColor(color)
                c.setAlpha(alpha)
                item = QGraphicsRectItem(QRectF(x, y0, max(bw * 0.85, 0.5), hh_))
                item.setBrush(QBrush(c))
                item.setPen(QPen(Qt.NoPen))
                r = trace.hit_bytes[i] + trace.miss_bytes[i]
                item.setToolTip(f"iteration {i + 1}: loop {lp}, level {lv}: DRAM read {trace.miss_bytes[i] / mb:.1f} MB, "
                                f"L2 hit {trace.hit_bytes[i] / mb:.1f} MB"
                                + (f" (hit rate {100 * trace.hit_bytes[i] / r:.0f}%)" if r else "")
                                + f", written back {trace.writeback_bytes[i] / mb:.1f} MB")
                scene.addItem(item)
            # level marker under the axis
            c = QColor(LEVEL_COLORS[(lv - 1) % len(LEVEL_COLORS)])
            item = QGraphicsRectItem(QRectF(x, base + low + SIZE / 80, max(bw * 0.85, 0.5), SIZE / 80))
            item.setBrush(QBrush(c))
            item.setPen(QPen(Qt.NoPen))
            scene.addItem(item)
        axis = QPen(QColor("#555555"), 1)
        axis.setCosmetic(True)
        scene.addLine(0, base, width, base, axis)
        # hit rate per iteration: a line over the bars, 100% at the top
        path = QPainterPath()
        for i in range(n):
            r = trace.hit_bytes[i] + trace.miss_bytes[i]
            rate = trace.hit_bytes[i] / r if r else 0.0
            point = (i * bw + bw * 0.42, base - hgt * rate)
            if i == 0:
                path.moveTo(*point)
            else:
                path.lineTo(*point)
        line = QGraphicsPathItem(path)
        pen = QPen(QColor("#222222"), 2)
        pen.setCosmetic(True)
        line.setPen(pen)
        line.setAcceptedMouseButtons(Qt.NoButton)
        line.setZValue(3)
        scene.addItem(line)
        self._text("100% hit", width + SIZE / 100, y - SIZE / 120, SIZE / 70, False, "#222222")
        self._text("0%", width + SIZE / 100, base - SIZE / 60, SIZE / 70, False, "#222222")
        frame = QPen(QColor("#222222"), 2)
        frame.setCosmetic(True)
        item = QGraphicsRectItem(QRectF(it * bw - 1, y - 2, bw + 2, hgt + low + SIZE / 40 + 4))
        item.setPen(frame)
        item.setAcceptedMouseButtons(Qt.NoButton)
        scene.addItem(item)
        self._text("levels: " + ", ".join(f"{l}" for l in range(1, trace.R + 1)) + " (colour strip)", 0,
                   base + low + SIZE / 30, SIZE / 60, False, "#555555")
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        if not self.fitted:
            self.fit()
            self.fitted = True

    def fit(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


def grid_summary(steps):
    return grid.schedule_summary(steps)


def schedule_html(N, R, operation, loops, steps, other, current, pipelined) -> str:
    """Explanation of the schedule view; other: the steps of the other schedule (comparison)"""
    s, o = grid.schedule_summary(steps), grid.schedule_summary(other)
    what = "A and B" if operation.startswith("encode") else "the root's C"
    mode = ("<b>Pipelined</b> (PIPELINED_LEVELS=ON): step s runs level l on row loop s - (l - 1), so a step "
            "holds the work of all levels whose loop exists; level l of loop k only needs level l - 1 of loop k "
            "(the same rows), done one step earlier." if pipelined else
            "<b>Original</b> (PIPELINED_LEVELS=OFF): loop after loop, level after level, one level per step.")
    return (f"<b>How the kernel runs {operation}</b> of {what}, N={N}, R={R}, {loops} row loops<br><br>"
            "bfs_kernel is one cooperative grid. A step hands out virtual blocks (32 x 4 threads, a warp covers "
            "256 halves of a row with half8) to the grid in rounds (grid stride), then every block waits at "
            "barrier.sync(). Each virtual block works on 4 rows of the current row loop of one level: in loop k "
            f"the rows k, k + {loops}, k + {2 * loops}, ... of every 2M x 2M group (the row ruler right of the matrix)."
            "<br><br>"
            + mode +
            f"<br><br>This schedule: <b>{s['barriers']}</b> barriers, {s['rounds']} rounds, "
            f"<b>{s['utilization'] * 100:.0f}%</b> busy<br>the other one: {o['barriers']} barriers, {o['rounds']} rounds, "
            f"{o['utilization'] * 100:.0f}% busy<br><br>"
            "<span style='color:#777'>Matrix: filled = written by a level in this step, dashed = read, frames = the "
            "2M x 2M groups; one colour per level. Right: the rows of a block, coloured by the level that works on "
            "them in this step (row r belongs to loop r mod loops). Schedule: hover a cell for its level, loop and blocks. Grid: the virtual blocks of the step "
            "by level, grey = blocks waiting at the barrier.</span>")


BOLD = " style='font-weight:bold'"


def l2_html(trace, it, comparison) -> str:
    """comparison: [(loops, Trace without hints, Trace with evict-first hints)]"""
    mb = 1024 * 1024
    loop, level = trace.iterations[it]
    rows = ""
    for lv in range(1, trace.R + 1):
        idx = [i for i, (_, l) in enumerate(trace.iterations) if l == lv]
        miss, hit = trace.miss_bytes[idx].sum(), trace.hit_bytes[idx].sum()
        rows += (f"<tr><td>{lv}</td><td align='right'>{miss / mb:,.0f}</td><td align='right'>{hit / mb:,.0f}</td>"
                 f"<td align='right'>{100 * hit / max(1, hit + miss):.0f}%</td></tr>")
    per_loop = (trace.hit_bytes + trace.miss_bytes).reshape(trace.loops, trace.R).sum(axis=1).max()
    return (f"<b>L2 during {trace.operation}</b> (original order: loop after loop, level after level), N={trace.N}, "
            f"R={trace.R}, {trace.loops} row loops, L2 {trace.l2_bytes / mb:g} MB<br><br>"
            f"Iteration {it + 1} of {len(trace.iterations)}: loop {loop}, level {level}: DRAM read "
            f"{trace.miss_bytes[it] / mb:.1f} MB, L2 hits {trace.hit_bytes[it] / mb:.1f} MB, written back "
            f"{trace.writeback_bytes[it] / mb:.1f} MB<br><br>"
            "Why the loops: level 2 reads what level 1 just wrote. One loop touches 1/loops of the rows of every "
            f"level, here up to {per_loop / mb:,.0f} MB per loop (all levels); when that fits in L2 the levels after "
            "the first find their data there (green), otherwise they read it from DRAM again (red). Try Loops = 1 and "
            "the tuned value.<br><br>"
            "<table border='1' cellspacing='0' cellpadding='3'><tr><th>level</th><th>DRAM read MB</th>"
            "<th>L2 hit MB</th><th>hit rate</th></tr>" + rows + "</table>"
            f"<br>total DRAM read {trace.miss_bytes.sum() / mb:,.0f} MB, written {(trace.writeback_bytes.sum() + trace.flush_bytes) / mb:,.0f} MB "
            f"({trace.flush_bytes / mb:,.0f} MB still dirty in L2 at the end), <b>hit rate {trace.hit_rate * 100:.0f}%</b><br><br>"
            "<b>Loop counts, without / with last-use hints</b>: the last access of a unit (rows this pass never "
            "touches again) as a last-use load (__ldlu, ld.global.lu) or an evict-first store (__stcs, st.global.cs), "
            "so it leaves L2 before data still needed"
            "<table border='1' cellspacing='0' cellpadding='3'><tr><th>loops</th><th>barriers</th>"
            "<th>DRAM read MB</th><th>DRAM write MB</th><th>hit rate</th></tr>"
            + "".join(f"<tr{BOLD if lp == trace.loops else ''}><td>{lp}</td>"
                      f"<td align='right'>{lp * trace.R}</td>"
                      f"<td align='right'>{a.miss_bytes.sum() / mb:,.0f} / {b.miss_bytes.sum() / mb:,.0f}</td>"
                      f"<td align='right'>{(a.writeback_bytes.sum() + a.flush_bytes) / mb:,.0f} / "
                      f"{(b.writeback_bytes.sum() + b.flush_bytes) / mb:,.0f}</td>"
                      f"<td align='right'>{a.hit_rate * 100:.0f}% / {b.hit_rate * 100:.0f}%</td></tr>"
                      for lp, a, b in comparison)
            + "</table><br>"
            "<span style='color:#777'>Model (gemm_l2/encode_l2.py): fully associative LRU of the L2 size, write-back; "
            f"a cache unit = the rows of one loop inside one of the {trace.cells} x {trace.cells} cells. Hover a cell or "
            "a bar for its bytes.</span>")


def passes_html(N, R, all_loops, G) -> str:
    """grid utilization of the four passes (loops: encode, bilinear, reconstruct, decode)"""
    rows = ""
    for name, op, loops in (("encode", "encode", all_loops[0]), ("forward", "forward", all_loops[1]),
                            ("reconstruct", "reconstruct", all_loops[2]), ("decode", "decode", all_loops[3])):
        its = grid.iterations(N, R, op, loops, G)
        s = grid.summary(its, loops)
        per = ", ".join(f"{i.utilization * 100:.0f}%" for i in its)
        color = "#c00" if s["utilization"] < 0.8 else "#000"
        rows += (f"<tr><td>{name}</td><td align='right'>{loops}</td><td align='right'>{s['barriers']}</td>"
                 f"<td align='right'>{s['rounds']}</td><td align='right' style='color:{color}'>"
                 f"{s['utilization'] * 100:.0f}%</td><td>{per}</td></tr>")
    return ("<br><br><b>Grid utilization of all passes</b> (G = %d blocks; L2 simulation tab loop counts)"
            "<table border='1' cellspacing='0' cellpadding='3'><tr><th>pass</th><th>loops</th><th>barriers</th>"
            "<th>rounds</th><th>busy</th><th>per level</th></tr>" % G + rows + "</table>"
            "<span style='color:#777'>busy: virtual blocks / (rounds x G); the rest of the block slots wait at "
            "grid barriers (gemm_l2/grid.py).</span>")


def info_html(N, R, operation, level, loops, loop, l2_bytes, stage) -> str:
    """Formula, legend and the bytes per level; stage: l2_sim.Stage of this transform"""
    leaf = N >> R
    read, written = encode.bytes_per_level(N, R, operation)
    matrices = 2 if operation.startswith("encode") else 1
    legend = "".join(f"<span style='color:{ROLE_COLORS[r]}'>&#9632;</span> {t}<br>"
                     for r, t in ROLE_TEXT.items() if r and (r in "+-d") == operation.startswith("encode"))
    legend += f"<span style='color:{CHANGED_COLOR}'>&#9632;</span> changed by an earlier level"
    rows = ""
    for lv, (dr, dw, req) in enumerate(stage.per_level, 1):
        per_loop = (read + written) * matrices / loops
        bold = ("<b>", "</b>") if lv == level else ("", "")
        rows += (f"<tr><td>{bold[0]}{lv}{bold[1]}</td><td align='right'>{N >> lv}</td>"
                 f"<td align='right'>{(1 << (lv - 1)) ** 2}</td>"
                 f"<td align='right'>{read * matrices / MB:,.0f} / {written * matrices / MB:,.0f}</td>"
                 f"<td align='right' style='color:{'#c00' if per_loop > l2_bytes else '#000'}'>"
                 f"{per_loop / MB:,.1f}</td>"
                 f"<td align='right'>{dr / MB:,.0f} / {dw / MB:,.0f}</td></tr>")
    which = "A and B (one kernel, both per thread)" if operation.startswith("encode") else "the root's C in S"
    return (f"<b>{operation}</b> of {which}, N={N}, R={R}, leaf {leaf}<br>"
            f"{encode.levels(R, operation)[0].formula}<br>on every 2M x 2M group a b / c d (thick frames), "
            "levels 1..R from the largest group down<br><br>" + legend +
            f"<br><br><b>Row loops: {loops}</b> (loop {loop} shown as lines: zoom in to see single rows): every "
            f"level, every group handles the rows {loop}, {loop + loops}, ... in loop {loop}, so one loop moves "
            f"1/{loops} of every level and the "
            "levels of one loop can reuse each other's rows in L2 "
            f"(L2 {l2_bytes / MB:g} MB).<br><br>"
            "<table border='1' cellspacing='0' cellpadding='3'><tr><th>level</th><th>M</th><th>groups</th>"
            "<th>read / write MB</th><th>per loop MB</th><th>DRAM read / write MB</th></tr>" + rows + "</table>"
            f"<br>DRAM {stage.total / MB:,.0f} MB for {stage.requested / MB:,.0f} MB requested "
            f"(minimum {stage.minimum / MB:,.0f} MB), gemm_l2/l2_sim.py. <span style='color:#777'>per loop: what one "
            "loop of the level reads and writes (red: more than L2).</span>")
