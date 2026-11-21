"""GPU-free launch diagnostics (moved from the C++ test_diagnostics).

    python py/diagnostics.py l2-warps [N R B] [--l2-mb 72] [--sms 128]
    python py/diagnostics.py select-nodes [N R]

l2-warps      warps and L2 footprint per forward level when B rows are processed per loop: every
              warp reads 2 rows of half8, level 0 reads A and B and writes S1..S6, deeper levels
              write S1..S6 of 7^l nodes. The GPU values (L2 size, SM count) are options instead
              of a device query (defaults: RTX 4090).
select-nodes  nodes of each level that no earlier pass covered, and the warps they need.

The GPU-bound diagnostics (warps, tune) stay in C++: ./bin/test_diagnostics.
"""

from __future__ import annotations

import argparse
import sys


def ubytes(mem: int) -> str:
    """core.hpp UBytes(): KB below 1 MB, MB below 1 GB, else GB."""
    mb = 1024 * 1024
    if mem < mb:
        return f"{mem / 1024.0:.6g}KB"
    if mem < mb * 1024:
        return f"{mem / 1024.0 / 1024:.6g}MB"
    return f"{mem / 1024.0 / 1024 / 1024:.6g}GB"


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def l2_warps(N: int, R: int, B: int, l2_bytes: int, sms: int, out=print):
    out(f"L2 Cache Size: {ubytes(l2_bytes)}")
    out(f"Using {sms} MP")
    leaf = N >> R
    if B > leaf:
        raise ValueError(f"B = {B} is larger than the leaf block {leaf}")
    if (B >> (R - 1)) == 0:
        raise ValueError(f"B = {B} is too small for R = {R} (B >> (R - 1) == 0)")

    def report(total_warps, total):
        warps_per_sm = div_up(total_warps, sms)
        out(f"Warps per SM: {warps_per_sm}, lost: {warps_per_sm * sms - total_warps}")
        return total

    H = N >> 1
    out(f"N = {N}, H ={H}")
    total = 0
    out("----- Layer 0")
    out(f"Rows: {B}")
    warps = H // 256
    out(f"Warps per row: {warps}")
    out("Nodes : 1")
    total_warps = B * warps // 2  # each warp reads 2 rows
    out(f"Total warps: {total_warps}")
    report(total_warps, total)
    in_mem = 2 * (total_warps * 512)  # read A and B, half8
    total += in_mem
    out(f"Input Memory: {ubytes(in_mem)}")
    out_mem = 6 * (total_warps * 512)
    total += out_mem
    out(f"Output Memory: {ubytes(out_mem)}")
    out(f"L2: {ubytes(total)}")
    if total > l2_bytes:
        out(f"L2 cache miss: {100.0 * (total - l2_bytes) / l2_bytes:g}%")
    for l in range(2, R + 1):
        out(f"----- Layer {l - 1}")
        BL = B >> (l - 1)
        out(f"Rows: {BL}")
        HL = N >> l
        out(f"N = {N}, H ={HL}")
        nodes = 7 ** (l - 1)
        warps = HL // 256
        total_warps = nodes * BL * warps // 2
        out(f"Warps per row: {warps}")
        out(f"Nodes : {nodes}")
        out(f"Total warps: {total_warps}")
        report(total_warps, total)
        out_mem = 6 * (total_warps * 512)
        total += out_mem
        out(f"Output Memory: {ubytes(out_mem)}")
        out(f"L2: {ubytes(total)}")
        if total > l2_bytes:
            out(f"L2 cache miss: {100.0 * (total - l2_bytes) / l2_bytes:g}%")


def select_nodes(N: int, R: int, out=print):
    """Nodes are numbered level by level (level l starts at (7^l - 1) / 6). For every pass, the
    nodes of that level not yet covered are followed down: each covered node marks the level below
    and continues with its children m1, m2 and m4 (the products that need no sums)."""
    width = 7 ** R // 6
    covered = [[False] * width for _ in range(R)]
    for pass_ in range(R):
        count = 7 ** pass_
        start = (count - 1) // 6
        missed = [col for col in range(start, start + count) if col < width and not covered[pass_][col]]
        for n in range(pass_ + 1, R + 1):
            W = N >> n  # W halves
            warps = W // 256
            out(f"Required warps: {len(missed) * warps}")
            new = []
            for front in missed:
                if n - 1 < R and front < width:
                    covered[n - 1][front] = True
                count_n = 7 ** n
                start1 = (count_n - 1) // 6
                start0 = start1 - count_n // 7
                idx = front - start0
                new += [start1 + idx * 7 + 0, start1 + idx * 7 + 1, start1 + idx * 7 + 3]
            missed = new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("l2-warps")
    p.add_argument("N", type=int, nargs="?", default=8192)
    p.add_argument("R", type=int, nargs="?", default=2)
    p.add_argument("B", type=int, nargs="?", default=64)
    p.add_argument("--l2-mb", type=float, default=72, help="L2 size in MB (RTX 4090: 72)")
    p.add_argument("--sms", type=int, default=128, help="number of SMs (RTX 4090: 128)")
    p = sub.add_parser("select-nodes")
    p.add_argument("N", type=int, nargs="?", default=8192)
    p.add_argument("R", type=int, nargs="?", default=4)
    args = parser.parse_args(argv)
    if args.mode == "l2-warps":
        l2_warps(args.N, args.R, args.B, int(args.l2_mb * 1024 * 1024), args.sms)
    else:
        select_nodes(args.N, args.R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
