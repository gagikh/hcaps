"""Which leaf products can run while the forward sums are still being formed?

A leaf product of the CAPS tree (7^R of them) multiplies two leaf blocks. Each operand is either a
quadrant chain of the encoded inputs A or B, or comes from the S1..S6 sums of some node. The sums
of the nodes of level l are formed by forward-pass step l + 1 ("sums level l + 1", levels 1 .. R),
so a product can start as soon as the sums of the deepest node its operands come from exist:

    stage(product) = 1 + max(level of the node whose S an operand is part of), 0 if both operands
                     come straight from A and B (needs only the encoder).

How operands are handed down (src/bfs.cpp initialize()): a node at level l with operands L, R gives
its children m1 = A11*B11, m2 = A12*B21, m3 = A21*S4, m4 = A22*B22, m5 = S1*S5, m6 = S2*S6,
m7 = S3*B12 (A.., B.. are quadrants of L, R; S.. its own sums). So a quadrant keeps the origin of
its parent's operand, and an S block has origin l.

Stage sizes: 3^R for stage 0 (only m1, m2, m4 on every level) and 4 * 7^(s-1) * 3^(R-s) for
stage s >= 1; they add up to 7^R.

GraphCAPS (src/graph_caps.*) already runs this: the encoder, then the sums one level at a time on a
second stream, and the leaf products in R + 1 batched groups, group s waiting only for sums level s.

timeline() is a simple model of that schedule against the serial one of BFS (all sums, then all
leaf GEMMs). Leaf GEMMs are limited by the tensor throughput. The sums are limited by DRAM
bandwidth, with different traffic in the two schedules:
  - serial (BFS): one kernel over all levels with row loops, the sums of level l are re-read from
    L2 by level l + 1, so DRAM sees A and B read once and every S written once (the measured
    BFS bilinear traffic on the 4090 matches this: N = 16384, R = 2: 1024 MB read, 2077 MB write);
  - overlapped (GraphCAPS): one kernel per level, so each level reads its parents' operands from
    DRAM again.
It assumes the two streams do not slow each other down, so the overlapped time is an optimistic
bound. The encoder runs before both schedules and is left out.
"""

from __future__ import annotations

from dataclasses import dataclass

MULTS = 7
HALF = 2  # bytes

# operand origin: -1 = A or B (encoded inputs), l >= 0 = the S area of a level l node
# children of a node (origin of L, origin of R, own level l) -> (origin L, origin R) of m1..m7
CHILD_ORIGINS = (
    lambda lo, ro, l: (lo, ro),   # m1 = A11 * B11
    lambda lo, ro, l: (lo, ro),   # m2 = A12 * B21
    lambda lo, ro, l: (lo, l),    # m3 = A21 * S4
    lambda lo, ro, l: (lo, ro),   # m4 = A22 * B22
    lambda lo, ro, l: (l, l),     # m5 = S1 * S5
    lambda lo, ro, l: (l, l),     # m6 = S2 * S6
    lambda lo, ro, l: (l, ro),    # m7 = S3 * B12
)
OPERANDS = ("A11 * B11", "A12 * B21", "A21 * S4", "A22 * B22", "S1 * S5", "S2 * S6", "S3 * B12")


@dataclass
class Product:
    index: int          # tree order: 7 * node + i
    node: int           # level R-1 node
    child: int          # 0..6: m1..m7 of that node
    path: tuple         # child index (0..6) on every level, root first
    origins: tuple      # (origin of L, origin of R), -1 = A / B
    stage: int          # 0 = right after the encoder, s = after sums level s

    @property
    def name(self):
        return f"P{self.node}.{self.child + 1}"

    def path_text(self):
        return ".".join(f"m{c + 1}" for c in self.path)


def products(R: int) -> list:
    """All 7^R leaf products in tree order with the stage they can start at."""
    if R < 1:
        raise ValueError("R must be at least 1")
    level = [((), -1, -1)]  # (path, origin L, origin R) of the nodes of the current level
    for l in range(R - 1):
        level = [(path + (c,),) + CHILD_ORIGINS[c](lo, ro, l) for path, lo, ro in level for c in range(MULTS)]
    out = []
    for node, (path, lo, ro) in enumerate(level):
        for c in range(MULTS):
            o = CHILD_ORIGINS[c](lo, ro, R - 1)
            out.append(Product(len(out), node, c, path + (c,), o, 1 + max(o)))
    return out


def stage_counts(R: int) -> list:
    counts = [0] * (R + 1)
    for p in products(R):
        counts[p.stage] += 1
    return counts


def expected_counts(R: int) -> list:
    """Closed form: 3^R, then 4 * 7^(s-1) * 3^(R-s)."""
    return [3 ** R] + [4 * MULTS ** (s - 1) * 3 ** (R - s) for s in range(1, R + 1)]


@dataclass
class Bar:
    lane: str           # "sums" or "gemm"
    stage: int
    start: float        # ms
    end: float


@dataclass
class Timeline:
    sums_ms: list       # sums level 1..R (index 0 unused, 0.0)
    gemm_ms: list       # leaf GEMM group per stage 0..R
    serial: list        # Bars
    overlapped: list    # Bars

    @property
    def serial_ms(self):
        return max(b.end for b in self.serial)

    @property
    def overlapped_ms(self):
        return max(b.end for b in self.overlapped)


def sums_bytes(N: int, level: int) -> int:
    """Minimal DRAM traffic of sums level `level` (1..R): the 7^(level-1) nodes of level level-1,
    M = N >> (level-1), each reads its L and R (2 M^2 halves) and writes S1..S6 (6 (M/2)^2)."""
    M = N >> (level - 1)
    return MULTS ** (level - 1) * (2 * M * M + 6 * (M // 2) * (M // 2)) * HALF


def sums_bytes_fused(N: int, R: int) -> int:
    """DRAM traffic of the BFS sums pass with L2 reuse between levels: A and B read once, the
    S1..S6 of every node of levels 0 .. R-1 written once."""
    writes = sum(MULTS ** l * 6 * (N >> (l + 1)) ** 2 for l in range(R))
    return (2 * N * N + writes) * HALF


def timeline(N: int, R: int, tflops: float, gbs: float) -> Timeline:
    """Serial vs overlapped schedule (see the module doc). tflops: effective leaf GEMM throughput,
    gbs: effective DRAM bandwidth of the sums kernels."""
    leaf = N >> R
    counts = stage_counts(R)
    gemm = [c * 2.0 * leaf ** 3 / (tflops * 1e12) * 1e3 for c in counts]
    sums = [0.0] + [sums_bytes(N, s) / (gbs * 1e9) * 1e3 for s in range(1, R + 1)]
    t = sums_bytes_fused(N, R) / (gbs * 1e9) * 1e3
    serial = [Bar("sums", 0, 0.0, t)]  # stage 0 = all levels in one pass
    for s in range(R + 1):
        serial.append(Bar("gemm", s, t, t + gemm[s]))
        t += gemm[s]
    overlapped, ready, t = [], [0.0], 0.0
    for s in range(1, R + 1):
        overlapped.append(Bar("sums", s, t, t + sums[s]))
        t += sums[s]
        ready.append(t)
    g = 0.0
    for s in range(R + 1):
        start = max(g, ready[s])
        overlapped.append(Bar("gemm", s, start, start + gemm[s]))
        g = start + gemm[s]
    return Timeline(sums, gemm, serial, overlapped)
