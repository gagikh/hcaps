"""Where can the leaf products go? Every leaf x leaf block of A, B and S, and who reads it.

A, B and S are split into blocks of leaf x leaf halves (leaf = N >> R). For every block the trace
records which leaf multiplication reads it directly as an operand. The leaf products of a level
R-1 node n are L_i * R_i with the m1..m7 pairs (A11*B11, A12*B21, A21*S4, A22*B22, S1*S5, S2*S6,
S3*B12) of n's own L, R and S; n's L and R are quadrants or S blocks of its ancestors, so a leaf
multiplication reads leaf blocks of A, of B, or of the S area of any level.

Structure (and what analyse() asserts): the four quadrants of a node's L go to m1..m4, the four of
its R to m1, m2, m4, m7, and its S1..S6 to m3, m5, m6, m7. So every block of a node's operands and
sums is handed to exactly one child, and by induction EVERY leaf block of A, B and S (except the
unused tail of S) is read by exactly one leaf multiplication: 2 * 7^R operand reads =
2 * 4^R blocks of A and B + the blocks of S. Nothing is unused before the leaf GEMMs; a block
becomes free once its only reader is done.

Pipeline around the products (src/bfs.cu, src/bfs.cpp):
  - leaf GEMMs: write the products; A and B (already encoded in place) and S are read;
  - leaf reconstruction: the 7 products of n -> n's product, written into n's S1..S4, for all
    level R-1 nodes concurrently; A and B are not needed any more;
  - intermediate reconstruction up to the root (C in the root's S1..S4), after that.
So a block may take a product of GEMM group k if its reader ran in an earlier group (< k), it is
not in the S1..S4 of a level R-1 node (overwritten by the leaf reconstruction while other
products still wait), or it is the unused tail of S. A product cannot overwrite its own operands:
a tiled GEMM reads every row of L and column of R for many output tiles.

GEMM groups: the leaf GEMMs as G consecutive batches over the level R-1 nodes in tree order
(G = 1: one batched GEMM, as today: no block is free before it ends, only the tail of S).
Using A, B or S for products needs code changes: per-product addresses (and ld = N halves) in the
batched GEMM and in the leaf reconstruction; A and B no longer hold the encoded inputs afterwards.

Coordinates: halves (x) and rows (y) of each buffer, as gemm_l2/layout.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gemm_l2.layout import HALF, MULTS, Layout, Rect, sums_blocks

# block status
TAIL = "tail"            # unused rows / columns of S: free from the start
RECON = "recon"          # S1..S4 of a level R-1 node: written by the leaf reconstruction
LIVE = "live"            # read by a leaf multiplication; free after the group of its reader

BUFFERS = ("A", "B", "S")


@dataclass
class Cell:
    status: str
    owner: str = ""          # "A", "B" or e.g. "L1.3 S5"
    last_group: int = -1     # GEMM group of the reading leaf multiplication (-1: never read)
    readers: int = 0         # leaf multiplications reading it (1 for every live block)
    reader: str = ""         # "P<node>.<i>": the leaf multiplication that reads it
    side: str = ""           # "L" or "R": the operand of that multiplication it belongs to
    product: str = ""        # with placement: "P<node>.<i>" stored here


@dataclass
class Reuse:
    lay: Layout
    groups: int
    cells: dict = field(default_factory=dict)        # (buf, cx, cy) -> Cell
    placed: dict = field(default_factory=dict)       # (node, i) -> (buf, cx, cy)
    placement_done: bool = False                     # place() ran

    @property
    def leaf(self):
        return self.lay.leaf

    @property
    def products(self):
        return MULTS ** self.lay.R

    def cell_rect(self, key) -> Rect:
        _, cx, cy = key
        return Rect(cx * self.leaf, cy * self.leaf, self.leaf, self.leaf)

    def count(self, status, buf=None):
        return sum(1 for k, c in self.cells.items() if c.status == status and (buf is None or k[0] == buf))

    def free_after(self, group):
        """Blocks that may take products of `group` (before placement)."""
        return [k for k, c in self.cells.items()
                if c.status == TAIL or (c.status == LIVE and c.last_group < group)]

    def cell_bytes(self):
        return self.leaf * self.leaf * HALF

    def saved_bytes(self):
        return len(self.placed) * self.cell_bytes()


def _quadrant(ref, i, j):
    buf, r = ref
    h = r.w // 2
    return buf, Rect(r.x + j * h, r.y + i * h, h, h)


def _children(L, R, S):
    """Operand pairs of m1..m7 (src/bfs.cpp initialize(), prepare_mult_buffers())."""
    A11, A12, A21, A22 = (_quadrant(L, i, j) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    B11, B12, B21, B22 = (_quadrant(R, i, j) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    s = {name: ("S", rect) for name, rect in S}
    return [(A11, B11), (A12, B21), (A21, s["S4"]), (A22, B22), (s["S1"], s["S5"]), (s["S2"], s["S6"]),
            (s["S3"], B12)]


def leaf_operands(lay: Layout):
    """Per level R-1 node (tree order): the 7 operand pairs ((buf, Rect), (buf, Rect)) of its products."""
    N, R = lay.N, lay.R
    level = [(("A", Rect(0, 0, N, N)), ("B", Rect(0, 0, N, N)))]  # (L, R) per node of the current level
    for l in range(R):
        nodes = lay.level_nodes(l)
        assert len(nodes) == len(level)
        pairs = [_children(L, Rr, sums_blocks(node)) for (L, Rr), node in zip(level, nodes)]
        if l == R - 1:
            return pairs
        level = [p for node_pairs in pairs for p in node_pairs]  # children in tree order: 7 * idx + i
    raise AssertionError("unreachable")


def analyse(lay: Layout, groups: int) -> Reuse:
    """Status of every leaf block of A, B and S for leaf GEMMs run in `groups` consecutive batches."""
    leaf = lay.leaf
    nodes = lay.level_nodes(lay.R - 1)
    groups = max(1, min(groups, len(nodes)))
    if lay.S.rows % leaf or lay.N % leaf:
        raise ValueError("S is not a whole number of leaf blocks")
    res = Reuse(lay, groups)
    blocks = lay.N // leaf
    for buf in ("A", "B"):
        for cy in range(blocks):
            for cx in range(blocks):
                res.cells[(buf, cx, cy)] = Cell(LIVE, buf)
    for cy in range(lay.S.rows // leaf):
        for cx in range(blocks):
            res.cells[("S", cx, cy)] = Cell(TAIL)

    def cells_of(buf, r: Rect):
        assert r.x % leaf == 0 and r.y % leaf == 0 and r.w % leaf == 0 and r.h % leaf == 0, r
        return [(buf, cx, cy) for cy in range(r.y // leaf, (r.y + r.h) // leaf)
                for cx in range(r.x // leaf, (r.x + r.w) // leaf)]

    for node in lay.nodes:  # owners of the S blocks
        for name, r in sums_blocks(node):
            recon = node.level == lay.R - 1 and name in ("S1", "S2", "S3", "S4")
            for k in cells_of("S", r):
                res.cells[k] = Cell(RECON if recon else LIVE, f"L{node.level}.{node.index} {name}")
    for j, pairs in enumerate(leaf_operands(lay)):  # readers
        group = j * groups // len(nodes)
        for i, pair in enumerate(pairs):
            for side, (buf, r) in zip("LR", pair):
                assert r.w == leaf and r.h == leaf
                for k in cells_of(buf, r):
                    c = res.cells[k]
                    assert c.status != TAIL, (k, "a leaf multiplication reads the unused tail")
                    c.readers += 1
                    c.reader = f"P{j}.{i + 1}"
                    c.side = side
                    c.last_group = max(c.last_group, group)
    return res


def place(res: Reuse) -> Reuse:
    """Greedy placement: the products of group k, node by node, into the first free blocks
    (S first, then A, then B; row by row)."""
    nodes = len(res.lay.level_nodes(res.lay.R - 1))
    order = {"S": 0, "A": 1, "B": 2}
    taken = set()
    res.placed = {}
    for c in res.cells.values():
        c.product = ""
    for k in range(res.groups):
        free = sorted((key for key in res.free_after(k) if key not in taken),
                      key=lambda t: (order[t[0]], t[2], t[1]))
        it = iter(free)
        members = [j for j in range(nodes) if j * res.groups // nodes == k]
        for j in members:
            for i in range(MULTS):
                key = next(it, None)
                if key is None:
                    break
                taken.add(key)
                res.placed[(j, i)] = key
                res.cells[key].product = f"P{j}.{i + 1}"
    res.placement_done = True
    return res


def group_choices(R: int):
    """Sensible group counts: whole subtrees, 1, 7, 49, ... up to one group per level R-1 node."""
    return [MULTS ** e for e in range(R)]


def summary(res: Reuse) -> dict:
    lay = res.lay
    cell = res.cell_bytes()
    out = {
        "groups": res.groups,
        "products": res.products,
        "placed": len(res.placed),
        "products_bytes": res.products * cell,
        "saved_bytes": res.saved_bytes(),
        "total_before": lay.total_bytes(),
        "total_after": lay.total_bytes() - res.saved_bytes(),
        "tail": res.count(TAIL),
        "recon": res.count(RECON),
        "read_more_than_once": sum(1 for c in res.cells.values() if c.readers > 1),
        "never_read": sum(1 for c in res.cells.values() if c.status != TAIL and c.readers == 0),
    }
    for buf in BUFFERS:
        out[f"{buf}_blocks"] = sum(1 for k in res.cells if k[0] == buf)
        out[f"{buf}_placed"] = sum(1 for key in res.placed.values() if key[0] == buf)
    return out
