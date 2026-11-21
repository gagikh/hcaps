"""L2 trace of the encoder / decoder of the original implementation (PIPELINED_LEVELS=OFF), for the
"L2" view of the Encoder tab.

    for row in 0 .. loops-1:            # row loop: rows row, row + loops, ... of the matrix
        for l in 1 .. R:                # level: every 2M x 2M group a b / c d, M = N >> l
            encode: read b, c, d, write d (A and B)      decode: read b, c, d, write b, c (C)
            grid barrier

The matrices are cut into cells x cells display cells; a cache unit is the rows of one row loop
inside one cell (rows row, row + loops, ... of the cell), so one iteration touches whole units.
The units go through a fully associative LRU of the L2 size (write-back, a full write needs no
read). Recorded per iteration (loop, level): the bytes read with an L2 hit, read from DRAM (miss),
written back to DRAM (dirty evictions), and per unit what happened and whether it is in L2 after.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

HALF = 2
# unit outcome in an iteration
NONE, HIT, MISS = 0, 1, 2
# unit state after an iteration
OUT, CLEAN, DIRTY = 0, 1, 2


@dataclass
class Trace:
    N: int
    R: int
    operation: str
    loops: int
    l2_bytes: int
    cells: int
    matrices: list          # names, "A", "B" or "C"
    iterations: list        # (loop, level)
    cell_bytes: int         # bytes of one display cell
    # per iteration and display cell [iterations, matrix, band, col], bytes:
    hit: np.ndarray         # read with an L2 hit
    miss: np.ndarray        # read from DRAM
    written: np.ndarray     # written
    resident: np.ndarray    # in L2 after the iteration
    dirty: np.ndarray       # of that, not yet written back
    hit_bytes: np.ndarray   # [iterations] totals
    miss_bytes: np.ndarray
    writeback_bytes: np.ndarray
    flush_bytes: int        # dirty bytes left in L2 at the end (written back eventually)
    evict_first: bool       # last accesses marked evict-first
    last_use: np.ndarray    # [iterations, matrix, band, col] bytes of last accesses (evict-first)

    @property
    def hit_rate(self):
        reads = self.hit_bytes.sum() + self.miss_bytes.sum()
        return self.hit_bytes.sum() / reads if reads else 0.0


def _loop_accesses(N, R, operation, cells, nm, row, loops, rpb):
    """[(level, (m, band, col), write)] of one row loop in the kernel's order: level after level; in a
    level the d rows ascending over all groups, A then B, b c d read then the written ones"""
    out = []
    for l in range(1, R + 1):
        mc = (N >> l) // rpb  # quadrant size in cells
        for gy in range(0, cells, 2 * mc):
            for band in range(mc):
                for gx in range(0, cells, 2 * mc):
                    for col in range(mc):
                        for m in range(nm):
                            b = (m, gy + band, gx + mc + col)        # a b: top right
                            c = (m, gy + mc + band, gx + col)        # c d: bottom left
                            d = (m, gy + mc + band, gx + mc + col)   # bottom right
                            cells_in = [(cl, False) for cl in (b, c, d)]
                            cells_in += [(cl, True) for cl in ((d,) if operation == "encode" else (b, c))]
                            for cl, write in cells_in:
                                if rpb < loops and (row - cl[1] * rpb) % loops >= rpb:
                                    continue  # the cell holds fewer rows than there are loops: none of this loop
                                out.append((l, cl, write))
    return out


def simulate(N: int, R: int, operation: str, loops: int, l2_bytes: int, cells: int = 32,
             evict_first: bool = False) -> Trace:
    """evict_first: an access that is the last one of its unit (rows never touched again: later
    loops work on other rows) puts the unit at the cold end of the LRU, as an L2::evict_first load or
    store would, so it leaves L2 before data that is still needed"""
    if operation not in ("encode", "decode"):
        raise ValueError("operation must be encode or decode")
    if (N >> R) % loops:
        raise ValueError(f"{loops} loops do not divide the leaf {N >> R}")
    cells = max(cells, 1 << R)  # quadrants of the deepest level must cover whole cells
    matrices = ["A", "B"] if operation == "encode" else ["C"]
    nm = len(matrices)
    rpb = N // cells           # rows (and columns) per cell
    rows_per_unit = max(1, rpb // loops)  # a unit: the rows of one row loop inside one cell
    unit_bytes = rows_per_unit * rpb * HALF
    lru: OrderedDict = OrderedDict()  # (m, band, col, loop) -> dirty
    used = 0
    shape = (nm, cells, cells)
    res = np.zeros(shape, np.int64)
    drt = np.zeros(shape, np.int64)
    iterations, H, Mi, W, RES, DRT, EF = [], [], [], [], [], [], []
    hits, misses, writebacks = [], [], []

    for row in range(loops):
        accesses = _loop_accesses(N, R, operation, cells, nm, row, loops, rpb)
        last = {}
        for i, (_, cl, _) in enumerate(accesses):
            last[cl] = i
        i = 0
        for l in range(1, R + 1):
            hit = np.zeros(shape, np.int64)
            miss = np.zeros(shape, np.int64)
            wr = np.zeros(shape, np.int64)
            ef = np.zeros(shape, np.int64)
            stat = [0, 0, 0]  # hit, miss, write back
            while i < len(accesses) and accesses[i][0] == l:
                _, key, write = accesses[i]
                u = key + (row,)
                if u in lru:
                    lru.move_to_end(u)
                    if write:
                        if not lru[u]:
                            drt[key] += unit_bytes
                        lru[u] = True
                        wr[key] += unit_bytes
                    else:
                        hit[key] += unit_bytes
                        stat[0] += unit_bytes
                else:
                    if write:
                        wr[key] += unit_bytes
                        drt[key] += unit_bytes
                    else:
                        miss[key] += unit_bytes
                        stat[1] += unit_bytes
                    lru[u] = write
                    res[key] += unit_bytes
                    used += unit_bytes
                if evict_first and last[key] == i:
                    lru.move_to_end(u, last=False)  # cold end: the next victim
                    ef[key] += unit_bytes
                while used > l2_bytes and len(lru) > 1:
                    v, dirty = lru.popitem(last=False)
                    used -= unit_bytes
                    res[v[:3]] -= unit_bytes
                    if dirty:
                        drt[v[:3]] -= unit_bytes
                        stat[2] += unit_bytes
                i += 1
            iterations.append((row, l))
            H.append(hit)
            Mi.append(miss)
            W.append(wr)
            EF.append(ef)
            RES.append(res.copy())
            DRT.append(drt.copy())
            hits.append(stat[0])
            misses.append(stat[1])
            writebacks.append(stat[2])
    flush = sum(unit_bytes for dirty in lru.values() if dirty)
    return Trace(N, R, operation, loops, l2_bytes, cells, matrices, iterations, rpb * rpb * HALF, np.array(H),
                 np.array(Mi), np.array(W), np.array(RES), np.array(DRT), np.array(hits, np.int64),
                 np.array(misses, np.int64), np.array(writebacks, np.int64), flush, evict_first, np.array(EF))
