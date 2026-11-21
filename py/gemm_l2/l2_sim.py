"""DRAM traffic of the whole CAPS pipeline through an L2 model, from the order the kernels touch memory.

Stages (src/bfs.cu, src/mma_gemm.cu, src/fused_strassen.cu), all on the buffers of gemm_l2/layout.py:

    encode       alternative basis transform of A and B in place, levels 1..R (alternative_basis.cuh)
    forward      S1..S6 of the nodes of levels 0..R-1 (bilinear_kernel); with the fused last level
                 only levels 0..R-2 (SKIP_LEAF_PARENT_SUMS)
    leaf         batched mma GEMM of the 7^R leaf products into P, in the tile order of mma_gemm_t
      or fused   batched fused R=1 kernel on the 7^(R-1) nodes of level R-1: 4 A + 4 B quadrant
                 panels per block, the node's C into its S area
    reconstruct  C of every node from its children's results (leaf: the 7 products in P; skipped
                 when the fused kernel wrote the level R-1 results)
    decode       inverse transform of the root's C (in S), in place

Model
    L2       one fully associative LRU cache of the GPU's L2 size, write-back, write-allocate without
             fill (only whole units are written). The cache hints of the kernels (__ldcg, __stcg,
             streaming loads) and the set-associative hashing of the real L2 are not modelled.
    passes   row loops: loop `row` of `loops` touches rows row, row + loops, ... of every region and
             nothing else, so loops share no data and all loops are alike: one loop is simulated and
             multiplied by `loops`. Inside a loop the levels run one after the other (grid barrier),
             nodes in order, rows ascending. A unit is g rows (of the loop's row class) x W halves,
             W = the narrowest region of the stage, g chosen so a unit is at most L2 / 256.
    GEMMs    blocks run in launch order (tile_order.launch_order(), product after product as
             blockIdx.z), sms x blocks_per_sm at a time: one wave, its blocks in step. Per k chunk of
             KC halves (a panel chunk at most L2 / 64) a wave reads the A and B panel chunks of its
             tiles once; the C tiles are
             written at the end. No-op blocks of the swizzle order take launch slots, as on the GPU.
    stages   every stage starts with an empty L2: what the previous stage left in L2 (at most one L2
             size) is not reused.

minimum = every byte the stage reads before writing it, read once, plus every byte it writes, written
once: the traffic of an infinite L2.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass

from .layout import MULTS, layout, sums_blocks
from .tile_order import CONFIGS, SWIZZLE_ORDER, Config, launch_order

MB = 1024 * 1024
HALF = 2
LQ = ("A11", "A12", "A21", "A22", "S1", "S2", "S3")  # left operand of m1..m7 (src/bfs.cpp)
RQ = ("B11", "B21", "S4", "B22", "S5", "S6", "B12")  # right operand
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


@dataclass(frozen=True)
class Gpu:
    name: str        # cudaDeviceProp::name, as in gen/tuning_table.hpp
    label: str
    sms: int
    l2_mb: float
    dram_gbs: float  # theoretical DRAM bandwidth


# as far as known: check against the spec sheet of the exact board
GPUS = (
    Gpu("NVIDIA GeForce RTX 4090", "RTX 4090", 128, 72, 1008),
    Gpu("NVIDIA GeForce RTX 3080 Ti Laptop GPU", "RTX 3080 Ti Laptop", 58, 4, 512),
    Gpu("NVIDIA GeForce RTX 5090", "RTX 5090", 170, 96, 1792),
    Gpu("NVIDIA A100-SXM4-40GB", "A100 40 GB", 108, 40, 1555),
    Gpu("NVIDIA H100 80GB HBM3", "H100 SXM", 132, 50, 3350),
)


# ------------------------------------------------------------------------------------------
# configurations and tuned parameters from the C++ sources
# ------------------------------------------------------------------------------------------

def fused_configs(path: str = "") -> list:
    """The configurations of src/fused_strassen.cu as tile_order.Config (BM, BN = quadrant tile).
    blocks_per_sm: 16 warps per SM for 16 x 16 warp tiles, else 8 (registers, ptxas -v), and at
    most 100 KB of shared memory per SM."""
    path = path or os.path.join(ROOT, "src", "fused_strassen.cu")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    configs = []
    pattern = r"using config(\d+) = config_t<(\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (true|false)>;"
    for m in re.finditer(pattern, text):
        index, wm, wn, tm, tn, bk, stages = (int(v) for v in m.groups()[:7])
        assert index == len(configs), f"config{index} out of order in {path}"
        bm, bn, warps = wm * tm, wn * tn, wm * wn
        smem_kb = stages * (4 * bm * bk + 4 * bk * bn) * HALF // 1024
        warps_per_sm = 16 if tm * tn == 16 * 16 else 8
        blocks = max(1, min(warps_per_sm // warps, 100 // smem_kb))
        configs.append(Config(bm, bn, bk, stages, smem_kb, blocks))
    if not configs:
        raise ValueError(f"no fused configurations in {path}")
    return configs


def tuned(gpu_name: str, N: int, R: int, path: str = "") -> dict:
    """Loop counts (GEMM_TUNED_BFS) and leaf configuration (GEMM_TUNED_LEAF, half accumulation) of
    gen/tuning_table.hpp for this GPU, else the "default" entries, else 1 loop / config 0 swizzle."""
    path = path or os.path.join(ROOT, "gen", "tuning_table.hpp")
    result = {"loops": (1, 1, 1, 1), "loops_from": "none", "mma": (0, SWIZZLE_ORDER), "mma_from": "none"}
    if not os.path.exists(path):
        return result
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for gpu in ("default", gpu_name):
        m = re.search(r'GEMM_TUNED_BFS\("%s", %d, %d, (\d+), (\d+), (\d+), (\d+)\)' % (re.escape(gpu), N, R), text)
        if m:
            result["loops"], result["loops_from"] = tuple(int(v) for v in m.groups()), gpu
    m = re.search(r'GEMM_TUNED_LEAF\("%s", %d, %d, 0, (\d+), (-?\d+)\)' % (re.escape(gpu_name), N >> R, 7 ** R),
                  text)
    if m:
        result["mma"], result["mma_from"] = (int(m.group(1)), int(m.group(2))), gpu_name
    return result


# ------------------------------------------------------------------------------------------
# regions and the tree
# ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """h x w halves of a buffer at (x, y): x in halves, y in rows"""
    buf: str
    x: int
    y: int
    w: int
    h: int

    def quad(self, q: str) -> "Region":
        h, w = self.h // 2, self.w // 2
        return Region(self.buf, self.x + (q[1] == "2") * w, self.y + (q[0] == "2") * h, w, h)

    @property
    def bytes(self):
        return self.w * self.h * HALF


@dataclass
class TreeNode:
    level: int
    index: int
    L: Region          # left operand, (N >> level) square
    R: Region          # right operand
    S: dict            # "S1".."S6" -> Region, (N >> (level + 1)) square
    C: Region          # the node's result (N >> level) square: its S1 S2 / S3 S4 area


def tree(N: int, R: int):
    """Nodes of levels 0..R-1 (level by level) with their operands, sums and result regions."""
    lay = layout(N, R)
    levels = []
    for node in lay.nodes:
        S = {name: Region("S", r.x, r.y, r.w, r.h) for name, r in sums_blocks(node)}
        size = N >> node.level
        C = Region("S", node.rect.x, node.rect.y, size, size)
        if node.level == 0:
            L, Rr = Region("A", 0, 0, N, N), Region("B", 0, 0, N, N)
        else:
            p = levels[node.level - 1][node.parent]
            left, right = LQ[node.child], RQ[node.child]
            L = p.L.quad(left[1:]) if left[0] == "A" else p.S[left]
            Rr = p.R.quad(right[1:]) if right[0] == "B" else p.S[right]
        if node.level == len(levels):
            levels.append([])
        levels[node.level].append(TreeNode(node.level, node.index, L, Rr, S, C))
    return lay, levels


def leaf_products(N: int, R: int, levels):
    """(A, B, C) regions of the 7^R leaf products in the order of the batched GEMM (prepare_mult_buffers)"""
    M = N >> R
    products = []
    for i, p in enumerate(levels[R - 1]):
        for c in range(MULTS):
            A = p.L.quad(LQ[c][1:]) if LQ[c][0] == "A" else p.S[LQ[c]]
            B = p.R.quad(RQ[c][1:]) if RQ[c][0] == "B" else p.S[RQ[c]]
            products.append((A, B, Region(f"P{i}", 0, c * M, M, M)))
    return products


# ------------------------------------------------------------------------------------------
# cache
# ------------------------------------------------------------------------------------------

class Lru:
    """Fully associative LRU, capacity in bytes, write-back, write-allocate without fill."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lines: OrderedDict = OrderedDict()  # key -> [bytes, dirty], least recently used first
        self.used = 0
        self.reads = self.writes = self.requested = 0
        self.min_reads = self.min_writes = 0
        self.accesses = 0
        self._seen, self._written = set(), set()

    def access(self, key, nbytes: int, write: bool, cold: bool = False):
        """cold: the last use (__ldlu / __stcs): the line goes to the cold end, the next victim"""
        self.accesses += 1
        self.requested += nbytes
        if key not in self._seen:
            self._seen.add(key)
            if not write:
                self.min_reads += nbytes
        if write and key not in self._written:
            self._written.add(key)
            self.min_writes += nbytes
        line = self.lines.get(key)
        if line is not None:
            self.lines.move_to_end(key, last=not cold)
            line[1] = line[1] or write
            return
        if not write:
            self.reads += nbytes
        self.lines[key] = [nbytes, write]
        if cold:
            self.lines.move_to_end(key, last=False)
        self.used += nbytes
        while self.used > self.capacity and len(self.lines) > 1:
            _, (b, dirty) = self.lines.popitem(last=False)
            self.used -= b
            if dirty:
                self.writes += b

    def flush(self):
        self.writes += sum(b for b, dirty in self.lines.values() if dirty)
        self.lines.clear()
        self.used = 0


@dataclass
class Stage:
    name: str
    reads: int          # DRAM bytes
    writes: int
    min_reads: int
    min_writes: int
    requested: int      # bytes the kernels ask for (L2 hits + misses)
    detail: str         # parameters of the simulation
    per_level: list     # passes: (reads, writes, requested) of every level (the final flush in the
                        # last level); GEMM stages: empty

    @property
    def total(self):
        return self.reads + self.writes

    @property
    def minimum(self):
        return self.min_reads + self.min_writes

    def ms(self, gbs: float) -> float:
        """time at the DRAM bandwidth (a lower bound for a memory-bound stage)"""
        return self.total / (gbs * 1e9) * 1e3


def _result(name, cache: Lru, scale: int, detail: str, marks: list) -> Stage:
    """marks: (reads, writes, requested) of the cache after each level (passes)"""
    cache.flush()
    per_level, previous = [], (0, 0, 0)
    for i, mark in enumerate(marks):
        if i == len(marks) - 1:
            mark = (cache.reads, cache.writes, cache.requested)
        per_level.append(tuple((a - b) * scale for a, b in zip(mark, previous)))
        previous = mark
    return Stage(name, cache.reads * scale, cache.writes * scale, cache.min_reads * scale,
                 cache.min_writes * scale, cache.requested * scale, detail, per_level)


def _pow2_floor(v: int) -> int:
    return 1 << max(0, int(v).bit_length() - 1)


# ------------------------------------------------------------------------------------------
# row-loop passes
# ------------------------------------------------------------------------------------------

@dataclass
class Task:
    """Rows 0..h-1 of every region of every segment, row group by row group; a segment's reads come
    before its writes"""
    h: int
    segments: list  # [(reads: [Region], writes: [Region])]


def simulate_pass(name: str, levels: list, loops: int, l2_bytes: int, hints: bool = False) -> Stage:
    """levels: [[Task]], one level after the other (grid barrier between them). hints: the last
    access of every unit in the pass is a last-use load / evict-first store (__ldlu / __stcs)"""
    regions = [r for lv in levels for t in lv for seg in t.segments for part in seg for r in part]
    if not regions:
        return Stage(name, 0, 0, 0, 0, 0, "nothing to do", [])
    W = min(r.w for r in regions)
    min_rows = min(t.h for lv in levels for t in lv) // loops
    if min_rows < 1:
        raise ValueError(f"{name}: {loops} loops for regions of {min(t.h for lv in levels for t in lv)} rows")
    g = _pow2_floor(max(1, min(min_rows, (l2_bytes // 256) // (W * HALF))))
    unit = g * W * HALF
    cache = Lru(l2_bytes)
    accesses, ends = [], []  # (key, write) of one loop; end index of every level
    for lv in levels:
        for t in lv:
            for k0 in range(0, t.h // loops, g):
                for reads, writes in t.segments:
                    for regs, write in ((reads, False), (writes, True)):
                        for r in regs:
                            y = r.y + k0 * loops
                            for x in range(r.x, r.x + r.w, W):
                                accesses.append(((r.buf, x, y), write))
        ends.append(len(accesses))
    last = {}
    if hints:
        for i, (key, _) in enumerate(accesses):
            last[key] = i
    marks, i = [], 0
    for end in ends:
        while i < end:
            key, write = accesses[i]
            cache.access(key, unit, write, hints and last[key] == i)
            i += 1
        marks.append((cache.reads, cache.writes, cache.requested))
    return _result(name, cache, loops, f"{loops} loops, unit {g} rows x {W} halves ({unit // 1024} KB), "
                                       f"{cache.accesses} accesses per loop", marks)


def encode_levels(N: int, R: int, matrices):
    """alternative_basis_transform_kernel with the encoder: per level l (M = N >> l) every 2M block
    a b / c d of A and B: read b, c, d, write d. Rows of all blocks of a block row together."""
    levels = []
    for l in range(1, R + 1):
        M = N >> l
        tasks = []
        for by in range(N // (2 * M)):
            segs = []
            for bx in range(N // (2 * M)):
                for m in matrices:
                    blk = Region(m.buf, m.x + bx * 2 * M, m.y + by * 2 * M, 2 * M, 2 * M)
                    b, c, d = blk.quad("12"), blk.quad("21"), blk.quad("22")
                    segs.append(([b, c, d], [d]))
            tasks.append(Task(M, segs))
        levels.append(tasks)
    return levels


def decode_levels(N: int, R: int, C: Region):
    """the decoder: read b, c, d, write b, c (one matrix, the root's C)"""
    levels = []
    for l in range(1, R + 1):
        M = N >> l
        tasks = []
        for by in range(N // (2 * M)):
            segs = []
            for bx in range(N // (2 * M)):
                blk = Region(C.buf, C.x + bx * 2 * M, C.y + by * 2 * M, 2 * M, 2 * M)
                b, c, d = blk.quad("12"), blk.quad("21"), blk.quad("22")
                segs.append(([b, c, d], [b, c]))
            tasks.append(Task(M, segs))
        levels.append(tasks)
    return levels


def forward_levels(levels_tree, count: int):
    """bilinear_kernel: level t reads the 4 + 4 quadrants of each node's operands, writes S1..S6"""
    out = []
    for nodes in levels_tree[:count]:
        tasks = []
        for n in nodes:
            reads = [n.L.quad(q) for q in ("11", "12", "21", "22")] + [n.R.quad(q) for q in ("11", "12", "21", "22")]
            writes = [n.S[f"S{i}"] for i in range(1, 7)]
            tasks.append(Task(n.S["S1"].h, [(reads, writes)]))
        out.append(tasks)
    return out


def reconstruct_levels(N: int, R: int, levels_tree, leaf: bool):
    """per loop: (leaf: the 7 products of every level R-1 node into its C) then levels R-1 .. 1:
    every node's C11..C22 (its S1..S4) from its 7 children's results"""
    out = []
    M = N >> R
    if leaf:
        tasks = []
        for i, p in enumerate(levels_tree[R - 1]):
            reads = [Region(f"P{i}", 0, c * M, M, M) for c in range(MULTS)]
            tasks.append(Task(M, [(reads, [p.S[f"S{k}"] for k in range(1, 5)])]))
        out.append(tasks)
    for l in range(R - 1, 0, -1):
        tasks = []
        for p in levels_tree[l - 1]:
            children = levels_tree[l][p.index * MULTS:(p.index + 1) * MULTS]
            tasks.append(Task(N >> l, [([c.C for c in children], [p.S[f"S{k}"] for k in range(1, 5)])]))
        out.append(tasks)
    return out


# ------------------------------------------------------------------------------------------
# GEMM stages
# ------------------------------------------------------------------------------------------

def simulate_gemm(name: str, products, size: int, cfg: Config, order: int, sms: int, l2_bytes: int,
                  quadrants: bool) -> Stage:
    """products: [(A, B, C)] regions. mma leaf (quadrants=False): size = M, tile (tm, tn) of C = A B.
    Fused (quadrants=True): A, B, C are the node's 2 size square operands and result, tiles of the
    size x size quadrants, every block reads the panels of all four A and all four B quadrants."""
    launch = launch_order(size, cfg, order)
    slots = int(launch.max()) + 1
    tiles = sorted((int(launch[tm, tn]), tm, tn) for tm in range(launch.shape[0]) for tn in range(launch.shape[1]))
    wave = sms * cfg.blocks_per_sm
    kc = _pow2_floor(max(cfg.BK, min(size, (l2_bytes // 64) // (max(cfg.BM, cfg.BN) * HALF))))
    kc = min(kc, size)
    qs = ((0, 0), (0, 1), (1, 0), (1, 1)) if quadrants else ((0, 0),)
    half = size if quadrants else 0
    a_bytes, b_bytes, c_bytes = cfg.BM * kc * HALF, kc * cfg.BN * HALF, cfg.BM * cfg.BN * HALF
    cache = Lru(l2_bytes)
    current, blocks = -1, []

    def run(blocks):
        for k in range(0, size, kc):
            seen = set()
            for z, tm, tn in blocks:
                A, B, _ = products[z]
                for qi, qj in qs:
                    key = (A.buf, A.y + qi * half + tm * cfg.BM, A.x + qj * half + k, "a")
                    if key not in seen:
                        seen.add(key)
                        cache.access(key, a_bytes, False)
                for qi, qj in qs:
                    key = (B.buf, B.y + qi * half + k, B.x + qj * half + tn * cfg.BN, "b")
                    if key not in seen:
                        seen.add(key)
                        cache.access(key, b_bytes, False)
        for z, tm, tn in blocks:
            C = products[z][2]
            for qi, qj in qs:
                cache.access((C.buf, C.y + qi * half + tm * cfg.BM, C.x + qj * half + tn * cfg.BN, "c"), c_bytes, True)

    for z in range(len(products)):
        for launch_id, tm, tn in tiles:
            w = (z * slots + launch_id) // wave
            if w != current:
                if blocks:
                    run(blocks)
                current, blocks = w, []
            blocks.append((z, tm, tn))
    if blocks:
        run(blocks)
    return _result(name, cache, 1, f"{cfg.BM}x{cfg.BN} tiles, waves of {wave} blocks, k chunks of {kc}, "
                                   f"{cache.accesses} accesses", [])


# ------------------------------------------------------------------------------------------
# pipeline
# ------------------------------------------------------------------------------------------

def pipeline(N: int, R: int, l2_bytes: int, sms: int, loops, leaf: str = "mma", mma_config: int = 0,
             mma_order: int = SWIZZLE_ORDER, fused_config: int = 8, fused_order: int = SWIZZLE_ORDER,
             alternative_basis: bool = True) -> list:
    """Stages of caps() (leaf "mma") or of the fused pipeline (leaf "fused"). loops: (encode,
    bilinear, reconstruct, decode) loop counts as in GEMM_TUNED_BFS."""
    if R < 1:
        raise ValueError("R must be at least 1")
    encode, bilinear, reconstruct, decode = loops
    lay, levels_tree = tree(N, R)
    root = levels_tree[0][0]
    stages = []
    if alternative_basis:
        stages.append(simulate_pass("encode", encode_levels(N, R, [root.L, root.R]), encode, l2_bytes))
    if leaf == "mma":
        stages.append(simulate_pass("forward", forward_levels(levels_tree, R), bilinear, l2_bytes))
        M = N >> R
        cfg = CONFIGS[mma_config]
        if M % cfg.BM or M % cfg.BN or M % cfg.BK:
            raise ValueError(f"mma config {mma_config} ({cfg.name}) does not divide the leaf {M}")
        stages.append(simulate_gemm(f"leaf mma #{mma_config}", leaf_products(N, R, levels_tree), M, cfg, mma_order,
                                    sms, l2_bytes, False))
        stages.append(simulate_pass("reconstruct", reconstruct_levels(N, R, levels_tree, True), reconstruct,
                                    l2_bytes))
    elif leaf == "fused":
        if R >= 2:
            stages.append(simulate_pass("forward", forward_levels(levels_tree, R - 1), bilinear, l2_bytes))
        cfg = fused_configs()[fused_config]
        H = N >> R
        if H % cfg.BM or H % cfg.BN or H % cfg.BK:
            raise ValueError(f"fused config {fused_config} does not divide the quadrant {H}")
        products = [(n.L, n.R, n.C) for n in levels_tree[R - 1]]
        stages.append(simulate_gemm(f"fused #{fused_config}", products, H, cfg, fused_order, sms, l2_bytes, True))
        if R >= 2:
            stages.append(simulate_pass("reconstruct", reconstruct_levels(N, R, levels_tree, False), reconstruct,
                                        l2_bytes))
    else:
        raise ValueError(f"leaf must be 'mma' or 'fused', not {leaf!r}")
    if alternative_basis:
        stages.append(simulate_pass("decode", decode_levels(N, R, root.C), decode, l2_bytes))
    return stages


def compare_orders(N: int, R: int, l2_bytes: int, sms: int, leaf: str, config: int, orders) -> list:
    """(order, Stage) of the GEMM stage alone for each tile order"""
    _, levels_tree = tree(N, R)
    out = []
    for order in orders:
        if leaf == "mma":
            cfg, size = CONFIGS[config], N >> R
            products, quadrants = leaf_products(N, R, levels_tree), False
        else:
            cfg, size = fused_configs()[config], N >> R
            products, quadrants = [(n.L, n.R, n.C) for n in levels_tree[R - 1]], True
        out.append((order, simulate_gemm(leaf, products, size, cfg, order, sms, l2_bytes, quadrants)))
    return out
