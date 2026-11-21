"""Grid utilization of the cooperative BFS passes (src/bfs.cu, src/alternative_basis.cuh).

bfs_kernel runs a fixed grid of G = blocks_per_sm x SMs / 4 blocks (PROCESS_GRID_DIVISOR = 4) of
32 x 4 threads (WARPS_IN_Y_DIR = 4 warps). Every iteration (one level of one row loop) hands out W
virtual blocks, block b taking b, b + G, b + 2G, ... (grid stride); then the whole grid waits at
barrier.sync(). Rounds = ceil(W / G); in the last round only W - (rounds - 1) G blocks work, the
others wait at the barrier: idle = rounds G - W block slots per iteration. When W < G a part of the
grid has nothing to do for the whole iteration.

Virtual blocks per iteration (T = half8: one warp covers 32 x 8 = 256 halves of a row):
    encode / decode   (N/2 / 256) x (N/2 / (4 loops))            every level the same
    forward level l   (M / 256) x ceil(7^(l-1) (N / loops >> l) / 4),   M = N >> l
    reconstruct leaf  (M / 256) x ceil(7^(R-1) (M / loops) / 4),        M = N >> R
    reconstruct l     (M / 256) x ceil(7^(l-1) (M / loops) / 4),        M = N >> l, l = R-1 .. 1
"""

from __future__ import annotations

from dataclasses import dataclass

DIM_X = 32 * 8      # halves per warp row (half8)
DIM_Y = 4           # warps per block (rows of a virtual block)
GRID_DIVISOR = 4    # PROCESS_GRID_DIVISOR in bfs.cu


@dataclass
class Iteration:
    level: int
    work: int        # virtual blocks W
    grid: int        # blocks G

    @property
    def rounds(self):
        return -(-self.work // self.grid)

    @property
    def idle(self):
        """block slots that wait at the barrier"""
        return self.rounds * self.grid - self.work

    @property
    def utilization(self):
        return self.work / (self.rounds * self.grid) if self.work else 0.0


def grid_blocks(sms: int, blocks_per_sm: int) -> int:
    return max(1, sms * blocks_per_sm // GRID_DIVISOR)


def _div_up(a, b):
    return -(-a // b)


def iterations(N: int, R: int, operation: str, loops: int, grid: int, levels: int = 0) -> list:
    """The iterations of one row loop (all loops are alike). operation: encode, decode, forward
    (levels: R, or R - 1 for the fused pipeline), reconstruct (leaf: True unless fused)"""
    out = []
    if operation in ("encode", "decode"):
        W = max(1, (N // 2) // DIM_X) * max(1, _div_up(N // 2, DIM_Y * loops))
        out = [Iteration(l, W, grid) for l in range(1, R + 1)]
    elif operation == "forward":
        rows = N // loops
        for l in range(1, (levels or R) + 1):
            M = N >> l
            W = max(1, M // DIM_X) * _div_up(7 ** (l - 1) * (rows >> l), DIM_Y)
            out.append(Iteration(l, W, grid))
    elif operation in ("reconstruct", "reconstruct (fused)"):
        if operation == "reconstruct":
            M = N >> R
            out.append(Iteration(R, max(1, M // DIM_X) * _div_up(7 ** (R - 1) * (M // loops), DIM_Y), grid))
        for l in range(R - 1, 0, -1):
            M = N >> l
            out.append(Iteration(l, max(1, M // DIM_X) * _div_up(7 ** (l - 1) * (M // loops), DIM_Y), grid))
    else:
        raise ValueError(f"unknown operation {operation!r}")
    return out


def summary(its: list, loops: int) -> dict:
    """totals over all loops"""
    slots = sum(i.rounds * i.grid for i in its) * loops
    work = sum(i.work for i in its) * loops
    return {"iterations": len(its) * loops, "barriers": len(its) * loops, "rounds": sum(i.rounds for i in its) * loops,
            "utilization": work / slots if slots else 0.0, "idle_slots": slots - work}


@dataclass
class Step:
    """One grid-barrier step: the (level, row loop, virtual blocks) parts handed out together"""
    index: int
    parts: list
    grid: int

    @property
    def work(self):
        return sum(w for _, _, w in self.parts)

    @property
    def rounds(self):
        return -(-self.work // self.grid)

    @property
    def utilization(self):
        return self.work / (self.rounds * self.grid) if self.work else 0.0


def schedule(N: int, R: int, operation: str, loops: int, grid: int, pipelined: bool, levels: int = 0) -> list:
    """The steps of a pass. Original: loop after loop, level after level, one level per step.
    Pipelined (PIPELINED_LEVELS): step s runs stage i (the i-th level of iterations()) on loop s - i."""
    stages = [(it.level, it.work) for it in iterations(N, R, operation, loops, grid, levels)]
    if pipelined:
        return [Step(s, [(lv, s - i, w) for i, (lv, w) in enumerate(stages) if 0 <= s - i < loops], grid)
                for s in range(loops + len(stages) - 1)]
    return [Step(k * len(stages) + i, [(lv, k, w)], grid) for k in range(loops) for i, (lv, w) in enumerate(stages)]


def schedule_summary(steps: list) -> dict:
    slots = sum(s.rounds * s.grid for s in steps)
    work = sum(s.work for s in steps)
    return {"barriers": len(steps), "rounds": sum(s.rounds for s in steps),
            "utilization": work / slots if slots else 0.0}
