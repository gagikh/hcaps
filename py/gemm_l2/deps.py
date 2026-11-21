"""Read/write dependency graph of the BFS pipeline below a start level, and its L2 traffic.

Unlike tree.biparted() (every leaf operand expanded down to the input blocks, i.e. all sums
formed on the fly), this follows the memory the BFS code really uses, at leaf-block granularity:

    - a node's operands are its parent's quadrants (views) or the parent's S1..S6 buffer
    - forward (bilinear kernel): reads the node's 4 + 4 quadrants, writes S1..S6
    - leaf multiplication m: reads its two operands, writes the product P (mult buffer)
    - reconstruct: reads the 7 children's results, writes C11..C22 into the node's own S1..S4
      (leaf level: the 7 products; above: the children's C, i.e. their S1..S4)

depth d = number of levels below the start level (start level = R - d). One subtree has
7^d leaf multiplications; subtrees share no memory, so a whole level is 7^(R - d) copies.

simulate() replays the ops through an LRU write-back cache (blocks = M x M halves) to count the
DRAM traffic of grouped execution: forward, GEMM and reconstruct of `group` subtrees at a time.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

LQ = ("A11", "A12", "A21", "A22", "S1", "S2", "S3")
RQ = ("B11", "B21", "S4", "B22", "S5", "S6", "B12")
SPOS = {"S1": (0, 0), "S2": (0, 1), "S3": (1, 0), "S4": (1, 1), "S5": (2, 0), "S6": (2, 1)}
QPOS = {"11": (0, 0), "12": (0, 1), "21": (1, 0), "22": (1, 1)}
# forward (src/bilinear.cuh): S = sum of signed operand quadrants
FWD_TERMS = {"S1": (("+", "A21"), ("+", "A22")), "S2": (("+", "A22"), ("-", "A12")),
             "S3": (("+", "A22"), ("-", "A11")), "S4": (("+", "B22"), ("-", "B11")),
             "S5": (("+", "B21"), ("+", "B22")), "S6": (("+", "B22"), ("-", "B12"))}
# reconstruct (src/reconstruct.cuh): C11..C22 (stored in S1..S4) = sum of signed products m1..m7
REC_TERMS = {"S1": (("+", 1), ("+", 2)), "S2": (("+", 5), ("-", 7)), "S3": (("+", 3), ("+", 6)),
             "S4": (("+", 5), ("+", 6), ("-", 2), ("-", 4))}


@dataclass(frozen=True)
class Region:
    """Square region of a buffer in leaf blocks: (buffer, row, column, size)."""
    buf: str
    r: int
    c: int
    n: int

    def quad(self, q: str) -> "Region":
        h = self.n // 2
        i, j = QPOS[q]
        return Region(self.buf, self.r + i * h, self.c + j * h, h)

    def blocks(self):
        return [(self.buf, self.r + i, self.c + j) for i in range(self.n) for j in range(self.n)]


@dataclass
class Op:
    name: str
    kind: str          # "fwd", "mult", "rec"
    level: int         # 0 = start level
    reads: list
    writes: list
    # fwd / rec: written block -> [(sign, read block)], the element-wise sum that forms it
    # (drawn as direct edges); mult: None
    sources: dict | None


def subtree_ops(depth: int, prefix: str = ""):
    """Ops of one subtree in BFS order: forward level by level, then the leaf products, then
    reconstruct from the deepest level up. Buffers: L, R (the start node's operands),
    S<path> (S1..S6 of the node at path), P<path> (leaf product)."""
    fwd, mults, rec = [], [], []

    def node(path, L, R, level):
        if level == depth:
            p = ("P" + prefix + path, 0, 0)
            mults.append(Op("m" + path, "mult", level, L.blocks() + R.blocks(), [p], None))
            return {(0, 0): p}
        h = L.n // 2
        S = "S" + prefix + path
        sop = {k: Region(S, v[0] * h, v[1] * h, h) for k, v in SPOS.items()}
        operand = {"A": L, "B": R}
        cells = [(i, j) for i in range(h) for j in range(h)]
        fsrc = {}
        for k, terms in FWD_TERMS.items():
            dst = sop[k]
            for i, j in cells:
                fsrc[(S, dst.r + i, dst.c + j)] = [
                    (sign, (lambda q: (q.buf, q.r + i, q.c + j))(operand[name[0]].quad(name[1:])))
                    for sign, name in terms]
        fwd.append(Op("fwd" + (path or ""), "fwd", level, L.blocks() + R.blocks(),
                      [b for k in SPOS for b in sop[k].blocks()], fsrc))
        children = []                       # result of m1..m7: (row, column) -> block
        for i in range(7):
            l = L.quad(LQ[i][1:]) if LQ[i][0] == "A" else sop[LQ[i]]
            r = R.quad(RQ[i][1:]) if RQ[i][0] == "B" else sop[RQ[i]]
            children.append(node(path + str(i + 1), l, r, level + 1))
        rsrc, result = {}, {}
        for k, terms in REC_TERMS.items():
            dst = sop[k]
            qi, qj = SPOS[k]                # S1..S4 hold C11, C12, C21, C22
            for i, j in cells:
                b = (S, dst.r + i, dst.c + j)
                rsrc[b] = [(sign, children[m - 1][(i, j)]) for sign, m in terms]
                result[(qi * h + i, qj * h + j)] = b
        reads = [b for c in children for b in c.values()]
        rec.append(Op("rec" + (path or ""), "rec", level, reads, list(rsrc), rsrc))
        return result

    n = 1 << depth
    node("", Region("L" + prefix, 0, 0, n), Region("R" + prefix, 0, 0, n), 0)
    fwd.sort(key=lambda o: o.level)                 # BFS: one kernel per level
    rec.sort(key=lambda o: -o.level)
    return fwd + mults + rec


# ------------------------------------------------------------------------------------------
# names and DOT
# ------------------------------------------------------------------------------------------

def _qpath(r, c, n):
    out = []
    while n > 1:
        n //= 2
        out.append(f"{r // n + 1}{c // n + 1}")
        r %= n
        c %= n
    return ".".join(out)


def block_name(block, depth: int) -> str:
    buf, r, c = block
    if buf[0] in "LR":
        return ("A" if buf[0] == "L" else "B") + _qpath(r, c, 1 << depth)
    if buf[0] == "S":
        path = buf[1:]
        h = 1 << (depth - len(path) - 1)
        k = (r // h) * 2 + (c // h) + 1
        sub = _qpath(r % h, c % h, h)
        return f"S{k}" + (f"<{path}>" if path else "") + ("." + sub if sub else "")
    return "P" + buf[1:]


KIND_COLOR = {"fwd": "#4e79a7", "mult": "#59a14f", "rec": "#e15759"}


def to_dot(ops, depth: int, rankdir: str = "LR") -> str:
    """Blocks are boxes (grey input, blue S/C, yellow product). Forward sums and reconstruct are
    direct block -> block edges (blue +, red -), one edge per term of the element-wise sum; the
    leaf multiplications are ellipses (operands in blue, product out grey). A block written again (C into
    S1..S4) is a new box joined by a dashed 'same memory' edge."""
    version = Counter()
    lines = ["digraph G {", f"rankdir={rankdir}; node [fontname=Helvetica,fontsize=10]; edge [arrowsize=.6];"]
    declared = set()

    def box(b, write):
        if write or not version[b]:
            version[b] += 1
        v = version[b]
        name = block_name(b, depth)
        node_id = f'"{name}#{v}"'
        if node_id not in declared:
            declared.add(node_id)
            label = name if v == 1 else re.sub(r"^S([1-4])", lambda m: "C" + ("11", "12", "21", "22")[int(m.group(1)) - 1], name)
            fill = "#dddddd" if b[0][0] in "LR" else ("#f2cf5b" if b[0][0] == "P" else "#a0cbe8")
            lines.append(f'{node_id} [shape=box,style=filled,fillcolor="{fill}",label="{label}"];')
            if v > 1:
                lines.append(f'"{name}#{v - 1}" -> {node_id} [style=dashed,color=gray,label="same memory",fontsize=8];')
        return node_id

    for op in ops:
        if op.sources is not None:
            # resolve every operand before any write of this op creates a new version
            terms = {w: [(sign, box(b, False)) for sign, b in src] for w, src in op.sources.items()}
            for w, src in terms.items():
                dst = box(w, True)
                for sign, s in src:
                    color = "#4e79a7" if sign == "+" else "#e15759"
                    lines.append(f'{s} -> {dst} [color="{color}"' + ('' if sign == "+" else ',label="-",fontcolor="#e15759"') + '];')
            continue
        lines.append(f'"{op.name}" [shape=ellipse,style=filled,fillcolor="{KIND_COLOR[op.kind]}",fontcolor=white];')
        for b in dict.fromkeys(op.reads):
            lines.append(f'{box(b, False)} -> "{op.name}" [color="#4e79a7"];')
        for b in dict.fromkeys(op.writes):
            lines.append(f'"{op.name}" -> {box(b, True)} [color="#555555"];')
    lines.append("}")
    return "\n".join(lines)


def summary(ops) -> dict:
    blocks = {b for op in ops for b in op.reads + op.writes}
    reads = Counter(b for op in ops for b in set(op.reads))
    return {
        "ops": len(ops),
        "mults": sum(1 for op in ops if op.kind == "mult"),
        "blocks": len(blocks),
        "block_reads": sum(len(set(op.reads)) for op in ops),
        "block_writes": sum(len(set(op.writes)) for op in ops),
        "reads_per_block": dict(sorted(Counter(reads.values()).items())),
    }


# ------------------------------------------------------------------------------------------
# L2 traffic
# ------------------------------------------------------------------------------------------

@dataclass
class Traffic:
    subtrees: int
    group: int
    cache_blocks: int
    reads: int = 0          # DRAM block reads (read misses)
    writes: int = 0         # DRAM block writes (dirty evictions + final flush)
    discarded: int = 0      # dirty scratch blocks dropped without writeback
    peak_group_blocks: int = 0  # distinct blocks one group touches
    per_op_kind: dict = field(default_factory=dict)

    @property
    def total(self):
        return self.reads + self.writes


def simulate(R: int, depth: int = 1, cache_blocks: int = 36, group: int = 1, ring: bool = False,
             discard: bool = False) -> Traffic:
    """DRAM traffic of the 7^(R - depth) subtrees below level R - depth, run `group` subtrees at a
    time (forward levels, leaf GEMMs, reconstruct levels of the group, then the next group).
    group >= number of subtrees is today's BFS: every stage over the whole level.

    ring:    S and product buffers of the group are reused by the next group (slot = subtree
             index % group); the start nodes' results C then go to separate output blocks.
    discard: a dirty block is dropped after its last read (discard.global.L2) instead of being
             written back on eviction; only never-read results (the start nodes' C) remain.
    Cache: fully associative LRU write-back, capacity in leaf blocks; full-block writes allocate
    without reading."""
    subtrees = 7 ** (R - depth)
    group = max(1, min(group, subtrees))
    ops = []
    for g0 in range(0, subtrees, group):
        members = range(g0, min(subtrees, g0 + group))
        per = []
        for t in members:
            slot = t % group if ring else t
            sub = subtree_ops(depth, prefix=f"<{t}>")
            for op in sub:
                # inputs L/R keep the subtree id, scratch (S, P) uses the ring slot
                def rename(b, t=t, slot=slot):
                    buf = b[0]
                    if buf[0] in "SP":
                        buf = re.sub(r"^([SP])<\d+>", rf"\g<1><{slot}>", buf)
                    return (buf,) + tuple(b[1:])
                reads = [rename(b) for b in op.reads]
                writes = [rename(b) for b in op.writes]
                if ring and op.kind == "rec" and op.level == 0:
                    writes = [("C<%d>" % t, b[1], b[2]) for b in op.writes]  # persistent result
                per.append((t, Op(op.name, op.kind, op.level, reads, writes, None)))
        # stage order inside the group: forward by level, mults, reconstruct deepest first
        def stage(item):
            op = item[1]
            return {"fwd": (0, op.level), "mult": (1, 0), "rec": (2, -op.level)}[op.kind]
        per.sort(key=stage)
        ops.append([op for _, op in per])

    flat = [op for g in ops for op in g]
    last_read = {}
    for i, op in enumerate(flat):
        for b in op.reads:
            last_read[b] = i

    t = Traffic(subtrees=subtrees, group=group, cache_blocks=cache_blocks)
    t.peak_group_blocks = max(len({b for op in g for b in op.reads + op.writes}) for g in ops)
    cache: OrderedDict = OrderedDict()  # block -> dirty, LRU -> MRU
    kind = Counter()

    def evict_one():
        b, dirty = cache.popitem(last=False)
        if dirty:
            t.writes += 1

    for i, op in enumerate(flat):
        for b in dict.fromkeys(op.reads):
            if b in cache:
                cache.move_to_end(b)
            else:
                t.reads += 1
                kind[op.kind] += 1
                if len(cache) >= cache_blocks:
                    evict_one()
                cache[b] = False
        for b in dict.fromkeys(op.writes):
            if b in cache:
                cache.move_to_end(b)
            elif len(cache) >= cache_blocks:
                evict_one()
            cache[b] = True
        if discard:
            # dirty data nobody reads again (S, products; C that is overwritten later) is dropped
            for b in dict.fromkeys(op.reads):
                if last_read[b] == i and cache.get(b):
                    del cache[b]
                    t.discarded += 1
    t.writes += sum(1 for dirty in cache.values() if dirty)
    t.per_op_kind = dict(kind)
    return t

