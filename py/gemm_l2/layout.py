"""Device memory layout of the BFS buffers, a port of BFS::create_intermediate_sums_layout() and
BFS::initialize() (src/bfs.cpp).

Buffers (fp16 data; the C++ code addresses them as float, i.e. pairs of halves):
  A, B     N x N halves each, the inputs
  S        one pitched buffer, N halves per row: the S1..S6 sums of every node of levels 0 .. R-1,
           each node an ROI of 3 x 2 sums (S1 S2 / S3 S4 / S5 S6) packed by find_roi()
  P        per node of the last level one continuous buffer of its 7 leaf products (7 M x M, M = N >> R)

Coordinates here are in halves (x, width) and rows (y, height); byte offsets are row * pitch + 2 x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MULTS = 7
HALF = 2  # bytes


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def empty(self):
        return self.w <= 0 or self.h <= 0


def find_roi(height: int, width: int, left: list) -> Rect:
    """find_roi() of src/bfs.cpp, in float units; `left` is the list of free rects, front first."""
    assert left
    front = left[0]
    if front.w == width:
        assert front.h >= height
        roi = Rect(front.x, front.y, width, height)
        front.h -= height
        front.y += height
        if front.h == 0:
            left.pop(0)
        return roi
    assert width < front.w
    if front.h < height:
        return Rect(0, 0, 0, 0)
    r1 = Rect(front.x, front.y, width, height)
    r2 = Rect(front.x + width, front.y, front.w - width, height)
    r3 = Rect(front.x, front.y + height, front.w, front.h - height)
    left.pop(0)
    if r3.h:
        left.insert(0, r3)
    if r2.h:
        left.insert(0, r2)
    return r1


def sums_rows(N: int, R: int) -> int:
    """Rows of the S buffer: create_intermediate_sums_layout()."""
    global_lds = N // 2
    rows = 0
    left: list = []
    count = 1
    for l in range(R):
        cols = (N >> l) // 2
        for _ in range(count):
            if not left:
                rows += 3 * cols
                left.append(Rect(0, 0, global_lds, 3 * cols))
            roi = find_roi(3 * cols, cols, left)
            if roi.empty():
                front = left[0]
                assert front.x == 0
                delta = 3 * cols - front.h
                front.h += delta
                rows += delta
        count *= MULTS
    return rows


@dataclass
class Node:
    level: int
    index: int          # within the level, tree order
    parent: int         # index in level - 1, -1 for the root
    child: int          # 0..6: m1..m7 of the parent
    rect: Rect          # in the S buffer, halves / rows
    operands: str       # what the node multiplies, e.g. "A21 * S4" of the parent


@dataclass
class Buffer:
    name: str
    rows: int
    cols: int           # halves
    pitch: int          # bytes per row

    @property
    def bytes(self):
        return self.rows * self.pitch


@dataclass
class Layout:
    N: int
    R: int
    A: Buffer
    B: Buffer
    S: Buffer
    products: list      # Buffer per node of the last level
    nodes: list = field(default_factory=list)   # Node, level by level
    free: list = field(default_factory=list)    # Rect (halves / rows) of S never used

    @property
    def leaf(self):
        return self.N >> self.R

    def level_nodes(self, level):
        return [n for n in self.nodes if n.level == level]

    def used_bytes(self):
        return sum(n.rect.w * n.rect.h for n in self.nodes) * HALF

    def total_bytes(self):
        return self.A.bytes + self.B.bytes + self.S.bytes + sum(p.bytes for p in self.products)


OPERANDS = ("A11 * B11", "A12 * B21", "A21 * S4", "A22 * B22", "S1 * S5", "S2 * S6", "S3 * B12")


def layout(N: int, R: int) -> Layout:
    """BFS::initialize(): the ROI of every node in S, in the order the C++ code assigns them."""
    if N & (N - 1) or N <= 0:
        raise ValueError("N must be a power of two")
    if R < 1:
        raise ValueError("R must be at least 1")
    if (N >> R) < 2:
        raise ValueError("N >> R is too small")
    global_lds = N // 2  # floats per row
    rows = sums_rows(N, R)
    pitch = global_lds * 4
    left = [Rect(0, 0, global_lds, rows)]

    def to_halves(r: Rect) -> Rect:
        return Rect(2 * r.x, r.y, 2 * r.w, r.h)

    nodes = []
    cols = N // 2
    roi = find_roi(3 * cols, cols, left)
    assert not roi.empty()
    nodes.append(Node(0, 0, -1, 0, to_halves(roi), "A * B"))
    previous = [nodes[0]]
    for l in range(1, R):
        M = N >> (l - 1)
        H = M // 2
        HH, WW = H // 2, H // 4
        current = []
        for parent in previous:
            for c in range(MULTS):
                roi = find_roi(3 * HH, 2 * WW, left)
                if roi.empty():
                    raise AssertionError(f"no room for level {l} node {len(current)} (C++ CV_Assert)")
                node = Node(l, len(current), parent.index, c, to_halves(roi), OPERANDS[c])
                current.append(node)
        nodes += current
        previous = current

    leaf = N >> R
    products = [Buffer(f"P{i}", MULTS * leaf, leaf, leaf * HALF) for i in range(len(previous))]
    return Layout(N, R, Buffer("A", N, N, N * HALF), Buffer("B", N, N, N * HALF), Buffer("S", rows, N, pitch),
                  products, nodes, [to_halves(r) for r in left if not r.empty()])


def sums_blocks(node: Node):
    """(name, Rect) of S1..S6 inside a node's ROI: S1 S2 / S3 S4 / S5 S6."""
    r = node.rect
    h, w = r.h // 3, r.w // 2
    return [(f"S{2 * i + j + 1}", Rect(r.x + j * w, r.y + i * h, w, h)) for i in range(3) for j in range(2)]
