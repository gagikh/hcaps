"""Write the images, graphs and generated sources (replaces the C++ test_generator).

    python py/generate.py [--out docs] [--cuh [DIR]]

Writes into --out (default <repo>/docs):
    biparted.dot                    leaf multiplication -> leaf block graph (R = 3)
    test.dot                        every leaf product (N = 8192, R = 3) as a tree of a+b / a-b
                                    nodes over the leaf blocks A_<y>_<x>, B_<y>_<x>
    deps_depth1.dot/.svg, deps_depth2.dot/.svg
                                    read/write dependency graph below the start level
                                    (.svg only when Graphviz is installed)
    tile_order_leaf<M>_config<c>.png
                                    tile orders of the mma leaf GEMM (gemm_l2/tile_order.py), all
                                    orders side by side, for leaf 1024 / 4096 with the 128x256 and
                                    256x128 configurations (--sms SMs per wave, default 58)
With --cuh [DIR] (default <repo>/gen) also depth_<R>_functions.cuh, R = 1..4: the operands of every leaf product
read straight from A and B (gemm_l2/codegen.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemm_l2 import codegen, deps, graphviz, tile_order  # noqa: E402
from gemm_l2.tree import biparted  # noqa: E402

MB = 1024 * 1024
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def biparted_dot(R: int) -> str:
    graph = biparted(R)
    C = 1 << R
    lines = ["digraph biparted {", "rankdir=LR; node [fontsize=10];"]
    for m, blocks in enumerate(graph):
        for b in blocks:
            name = f"A{b // C}_{b % C}" if b < C * C else f"B{(b - C * C) // C}_{(b - C * C) % C}"
            lines.append(f'"{name}" -> "m{m}";')
    lines.append("}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO, "docs"))
    parser.add_argument("--sms", type=int, default=58, help="SMs per wave of the tile order images (RTX 3080 Ti "
                                                            "Laptop: 58, RTX 4090: 128)")
    parser.add_argument("--cuh", nargs="?", const=os.path.join(REPO, "gen"),
                        help="also write depth_<R>_functions.cuh into this folder (default <repo>/gen)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)  # noqa: E731
    N = 8192

    with open(out("biparted.dot"), "w") as f:
        f.write(biparted_dot(3))
    with open(out("test.dot"), "w") as f:
        f.write(codegen.tree_dot(N, 3))

    for depth in (1, 2):
        text = deps.to_dot(deps.subtree_ops(depth), depth)
        with open(out(f"deps_depth{depth}.dot"), "w") as f:
            f.write(text)
        try:
            with open(out(f"deps_depth{depth}.svg"), "wb") as f:
                f.write(graphviz.render(text))
        except RuntimeError as e:
            print(f"deps_depth{depth}.svg skipped: {e}")

    for M, config in ((1024, 0), (4096, 0), (4096, 1)):
        path = out(f"tile_order_leaf{M}_config{config}.png")
        cv2.imwrite(path, tile_order.compare_orders_image(M, tile_order.CONFIGS[config], args.sms, 2 * M))
        print(path)

    if args.cuh:
        os.makedirs(args.cuh, exist_ok=True)
        for R in (1, 2, 3, 4):
            path = os.path.join(args.cuh, f"depth_{R}_functions.cuh")
            with open(path, "w") as f:  # text mode: CRLF on Windows, like the C++ generator
                f.write(codegen.forward_pass_header(R))
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
