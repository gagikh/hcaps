"""pytest py/tests"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import hashlib  # noqa: E402

from gemm_l2 import codegen, deps  # noqa: E402
from gemm_l2.tree import biparted, coefficients, mult_count  # noqa: E402

MB = 1024 * 1024


def test_graph_shape():
    for R in (1, 2, 3):
        graph = biparted(R)
        assert len(graph) == mult_count(R)
        C = 1 << R
        assert {b for ops in graph for b in ops} == set(range(2 * C * C))


def test_graph_r1():
    # m1..m7 in the C x C = 2 x 2 grid: A ids 0..3, B ids 4..7
    assert biparted(1) == [[0, 4], [1, 6], [2, 4, 7], [3, 7], [2, 3, 6, 7], [1, 3, 5, 7], [0, 3, 5]]
    signs = coefficients(1)
    assert signs[6][0] == {3: 1, 0: -1}   # S3 = A22 - A11
    assert signs[2][1] == {7: 1, 4: -1}   # S4 = B22 - B11


def test_deps_one_parent():
    ops = deps.subtree_ops(1)
    s = deps.summary(ops)
    # 8 quadrants + S1..S6 + 7 products (C reuses S1..S4)
    assert (s["ops"], s["mults"], s["blocks"]) == (9, 7, 21)
    assert s["reads_per_block"] == {1: 13, 2: 8}


def test_deps_traffic():
    # R = 3, M = 1024 on a 72 MB L2 (36 blocks): today ~46 blocks per parent, one parent at a time
    # with ring buffers and discard: 8 reads + 4 writes
    today = deps.simulate(3, 1, 36, group=49)
    grouped = deps.simulate(3, 1, 36, group=1, ring=True, discard=True)
    assert round(today.total / today.subtrees) == 46
    assert grouped.total / grouped.subtrees == 12


def test_codegen_matches_cpp():
    # md5 of the C++ generator's output (LF line ends): archive/gen/depth_2_functions.cuh,
    # archive/gen/depth_3_functions.cuh and docs/test.dot
    md5 = lambda text: hashlib.md5(text.encode()).hexdigest()  # noqa: E731
    assert md5(codegen.forward_pass_header(2)) == "bdce4bef544f0d0a555e531f81da0ac1"
    assert md5(codegen.forward_pass_header(3)) == "a3ed31a4a5ce95779d59258964055632"
    assert md5(codegen.tree_dot(8192, 3)) == "9d28dc4b0c6176a32022e8698df7304c"


def test_codegen_operands_match_tree():
    # the expression trees read the same leaf blocks as tree.biparted()
    R, N = 2, 1024
    C, M = 1 << R, N >> R

    def blocks(node, side):
        if node.leaf:
            return {node.x // M + C * (node.y // M) + (C * C if side == "B" else 0)}
        return blocks(node.L, side) | blocks(node.R, side)

    graph = biparted(R)
    for m, node in enumerate(codegen.build(N, R)):
        assert sorted(blocks(node.L, "A")) + sorted(blocks(node.R, "B")) == graph[m]


def test_plots_from_results(tmp_path):
    # the CSV format of tst/results.cpp -> PNGs
    import plots
    data = tmp_path / "data"
    data.mkdir()
    head = "# kind: {}\n# title: {}\n# y_label: energy (mJ)\n# device: test\n"
    rows = "".join(f"{t},{rng},{R},{N},1,3,2,0,0\n" for t in ("performance", "precision") for rng in (1, 2)
                   for R in (1, 2) for N in (1024, 2048, 4096))
    for name in ("cublas", "caps_cublas"):
        (data / f"{name}.csv").write_text(head.format("stats", name) + "type,range,R,N,min,max,avg,std,rel\n" + rows)
    (data / "relative_costs.csv").write_text(head.format("series", "Relative Costs") + "series,N,value\n"
                                             "CUBLAS,1024,1\nCUBLAS,2048,8\nCAPS R=1,2048,7\n")
    written = plots.plot_all(str(data), str(tmp_path))
    names = {os.path.basename(p) for p in written}
    assert "Costs.png" in names and "Compare_caps_vs_cublas.png" in names
    assert len(written) == 1 + 1 + 2 + 2 + 4 + 1 + 1 + 4 + 2  # series, its ratio to cuBLAS, best x2, per R, ...
    assert "Relative_Costs.png" in names  # the ratio to cuBLAS
    assert all(os.path.getsize(p) > 0 for p in written)


def test_diagnostics_l2_warps():
    import diagnostics
    lines = []
    diagnostics.l2_warps(8192, 2, 64, 72 << 20, 128, out=lines.append)
    assert "Total warps: 512" in lines and "Nodes : 7" in lines


def test_gpu_folders():
    import plots
    assert plots.gpu_folder("NVIDIA GeForce RTX 4090") == "4090"
    assert plots.gpu_folder("NVIDIA GeForce RTX 5090") == "5090"
    assert plots.gpu_folder("NVIDIA GeForce RTX 3080 Ti Laptop GPU") == "3080ti-laptop"
    assert plots.gpu_folder("NVIDIA H100 80GB HBM3") == "h100"
    assert plots.gpu_folder("NVIDIA H200") == "h200"
    assert plots.gpu_folder("NVIDIA A100-SXM4-80GB") == "a100-sxm"


def test_store_run(tmp_path):
    import plots
    data = tmp_path / "data"
    data.mkdir()
    (data / "relative_costs.csv").write_text("# kind: series\n# title: Relative Costs\n# y_label: mJ\n"
                                             "# device: NVIDIA GeForce RTX 4090\n# run: 2026-09-26_14-05-33\n"
                                             "series,N,value\nCUBLAS,1024,1\nCUBLAS,2048,8\n")
    folder = plots.store(str(data), str(tmp_path / "repo"))
    assert folder == str(tmp_path / "repo" / "4090" / "2026-09-26_14-05-33")
    assert os.path.isfile(os.path.join(folder, "data", "relative_costs.csv"))
    assert os.path.isfile(os.path.join(folder, "Costs.png"))  # cuBLAS only: no ratio graph
    with pytest.raises(RuntimeError):
        plots.store(str(data), str(tmp_path / "repo"))  # the same run twice


@pytest.mark.parametrize("N,R", [(1024, 1), (1024, 2), (1024, 3), (2048, 4), (4096, 5)])
def test_memory_layout(N, R):
    # the port of BFS::initialize(): every node fits (the C++ CV_Assert), no ROI overlaps, S is covered
    import numpy as np
    from gemm_l2.layout import layout
    lay = layout(N, R)
    assert [len(lay.level_nodes(l)) for l in range(R)] == [7 ** l for l in range(R)]
    assert len(lay.products) == 7 ** (R - 1)
    g = 64  # every ROI is a multiple of N >> R >> 1 rows / halves; count on a coarse grid
    unit = max(1, (N >> R) // 2 // g) if (N >> R) // 2 >= g else 1
    mask = np.zeros((lay.S.rows // unit, lay.S.cols // unit), np.int32)
    for n in lay.nodes:
        r = n.rect
        assert r.h == 3 * (N >> (n.level + 1)) and r.w == 2 * (N >> (n.level + 1))
        mask[r.y // unit:(r.y + r.h) // unit, r.x // unit:(r.x + r.w) // unit] += 1
    for f in lay.free:
        mask[f.y // unit:(f.y + f.h) // unit, f.x // unit:(f.x + f.w) // unit] += 1
    assert (mask == 1).all()
    if (N, R) == (1024, 2):  # S rows of create_intermediate_sums_layout()
        assert lay.S.rows == 4608


def test_tile_orders():
    # every tile gets exactly one block; group 1 is row by row, COLUMN_ORDER column by column,
    # the swizzle walks strips of 8 tile columns (CUTLASS GemmIdentityThreadblockSwizzle<8>)
    import numpy as np
    from gemm_l2 import tile_order as t
    cfg = t.CONFIGS[0]  # 128 x 256
    for M in (256, 1024, 4096):
        for order in t.ORDERS + (3,):
            launch = t.launch_order(M, cfg, order)
            assert sorted(launch.flatten()) == list(range(launch.size)) or order == t.SWIZZLE_ORDER
            assert len(set(launch.flatten())) == launch.size
    rows = t.launch_order(1024, cfg, 1)
    assert (rows == np.arange(rows.size).reshape(rows.shape)).all()
    cols = t.launch_order(1024, cfg, t.COLUMN_ORDER)
    assert (cols == np.arange(cols.size).reshape(cols.shape[::-1]).T).all()
    sw = t.launch_order(4096, cfg, t.SWIZZLE_ORDER)  # 32 x 16 tiles: strips of 8 columns
    assert list(sw[0, :9]) == [0, 1, 2, 3, 4, 5, 6, 7, 256] and sw[1, 0] == 8
    s = t.wave_stats(4096, cfg, 1, 58, 8192)
    assert (s.a_panels, s.b_panels, s.waves) == (4, 16, 9)


def test_tile_order_image():
    from gemm_l2 import tile_order as t
    image, s = t.tile_order_image(1024, t.CONFIGS[4], t.SWIZZLE_ORDER, 58, 2048)
    assert image.ndim == 3 and s.wave_blocks == 116
    assert t.compare_orders_image(1024, t.CONFIGS[0]).shape[0] > 0


def test_l2_sim_configs_and_tuning():
    from gemm_l2 import l2_sim
    fused = l2_sim.fused_configs()  # parsed from src/fused_strassen.cu
    assert (fused[8].BM, fused[8].BN, fused[8].BK, fused[8].smem_kb, fused[8].blocks_per_sm) == (64, 128, 32, 96, 1)
    assert (fused[17].BM, fused[17].BN, fused[17].blocks_per_sm) == (32, 64, 2)
    t = l2_sim.tuned("no such GPU", 16384, 2)
    assert t["loops_from"] in ("default", "none") and len(t["loops"]) == 4


def test_l2_sim_regions():
    from gemm_l2 import l2_sim
    lay, levels = l2_sim.tree(8192, 2)
    root = levels[0][0]
    assert root.L == l2_sim.Region("A", 0, 0, 8192, 8192) and root.S["S6"].h == 4096
    m3 = levels[1][2]   # m3 = A21 * S4 of the root
    assert m3.L == root.L.quad("21") and m3.R == root.S["S4"]
    products = l2_sim.leaf_products(8192, 2, levels)
    assert len(products) == 49 and products[4][0] == levels[1][0].S["S1"]


def test_l2_sim_minimum_with_large_l2():
    # an L2 larger than everything: DRAM traffic = minimum in every stage
    from gemm_l2 import l2_sim
    stages = l2_sim.pipeline(2048, 2, 1 << 40, 58, (1, 1, 1, 1), "mma", 0, -1, 8, -1, True)
    assert [s.name for s in stages] == ["encode", "forward", "leaf mma #0", "reconstruct", "decode"]
    for s in stages:
        assert s.total == s.minimum, s.name
    forward = stages[1]
    assert forward.min_reads == 2 * 2048 * 2048 * 2 and forward.min_writes == 6 * 1024 * 1024 * 2 + 42 * 512 * 512 * 2
    leaf = stages[2]
    assert leaf.min_writes == 49 * 512 * 512 * 2


def test_l2_sim_small_l2_and_orders():
    from gemm_l2 import l2_sim, tile_order
    # one row loop of a pass that does not fit: the level 2 reads of the level 1 sums miss
    big = l2_sim.pipeline(4096, 2, 4 << 20, 58, (1, 1, 1, 1), "mma", 0, -1, 8, -1, False)
    tuned = l2_sim.pipeline(4096, 2, 4 << 20, 58, (1, 64, 64, 1), "mma", 0, -1, 8, -1, False)
    assert big[0].total > big[0].minimum and tuned[0].total == tuned[0].minimum
    # fused R = 1, 8192 on a 72 MB L2: column order re-reads A, the swizzle much less
    res = dict(l2_sim.compare_orders(8192, 1, 72 << 20, 128, "fused", 8,
                                     (tile_order.SWIZZLE_ORDER, tile_order.COLUMN_ORDER)))
    assert res[tile_order.COLUMN_ORDER].total > 2 * res[tile_order.SWIZZLE_ORDER].total
    fused = l2_sim.pipeline(8192, 2, 72 << 20, 128, (1, 1, 1, 1), "fused", 0, -1, 8, -1, True)
    assert [s.name for s in fused] == ["encode", "forward", "fused #8", "reconstruct", "decode"]


def test_encode_expressions():
    from gemm_l2 import encode
    s = encode.expressions(1, "encode A")
    assert encode.expression_text(s[1][(1, 1)], "A", 1) == "A12 + A22 − A21"
    s = encode.expressions(1, "encode B")
    assert encode.expression_text(s[1][(1, 1)], "B", 1) == "B21 + B22 − B12"
    s = encode.expressions(1, "decode C")
    assert [encode.expression_text(s[1][k], "C", 1) for k in ((0, 1), (1, 0))] == ["C12 − C22", "C22 − C21"]
    # two levels: the negative terms survive (Counter + would drop them)
    s = encode.expressions(2, "encode A")
    assert len(s[2][(3, 3)]) == 9 and sum(1 for c in s[2][(3, 3)].values() if c < 0) == 4
    roles = encode.levels(2, "encode A")[1].roles
    assert (roles[(0, 1)], roles[(1, 0)], roles[(1, 1)], roles[(0, 0)]) == ("+", "-", "d", "")


def test_grid_utilization():
    from gemm_l2 import grid
    G = grid.grid_blocks(128, 12)  # 384
    enc = grid.iterations(16384, 2, "encode", 32, G)
    assert [(i.work, i.rounds, i.idle) for i in enc] == [(2048, 6, 256)] * 2
    fwd = grid.iterations(16384, 2, "forward", 256, G)
    assert [(i.work, i.rounds) for i in fwd] == [(256, 1), (448, 2)]  # level 1: a third of the grid idle
    s = grid.summary(fwd, 256)
    assert s["barriers"] == 512 and abs(s["utilization"] - 704 / 1152) < 1e-9
    rec = grid.iterations(16384, 2, "reconstruct", 128, G)
    assert [i.level for i in rec] == [2, 1]


def test_encode_l2_trace_matches_l2_sim():
    from gemm_l2 import encode_l2, l2_sim
    MB = 1 << 20
    for N, R, loops, l2 in ((2048, 3, 256, 1 * MB), (8192, 2, 4, 4 * MB), (4096, 2, 1, 8 * MB)):
        tr = encode_l2.simulate(N, R, "encode", loops, l2, 32)
        _, lv = l2_sim.tree(N, R)
        st = l2_sim.simulate_pass("encode", l2_sim.encode_levels(N, R, [lv[0][0].L, lv[0][0].R]), loops, l2)
        assert tr.miss_bytes.sum() == st.reads
        assert tr.writeback_bytes.sum() + tr.flush_bytes == st.writes
        assert len(tr.iterations) == loops * R


@pytest.mark.parametrize("N,R", [(4096, 1), (8192, 2), (8192, 3), (16384, 4)])
def test_products_in_s(N, R):
    from gemm_l2.layout import layout
    from gemm_l2.reuse import LIVE, RECON, TAIL, analyse, group_choices, place

    lay = layout(N, R)
    nodes = 7 ** (R - 1)  # level R - 1 nodes
    for groups in group_choices(R):
        res = place(analyse(lay, groups))
        # every leaf block of A, B and of the nodes' S areas is the operand of exactly one leaf
        # multiplication; the tail of S of none
        assert all(c.readers == 1 for c in res.cells.values() if c.status != TAIL)
        assert sum(1 for k in res.cells if k[0] in "AB") == 2 * 4 ** R
        assert all(c.readers == 0 for c in res.cells.values() if c.status == TAIL)
        for (j, i), key in res.placed.items():
            c = res.cells[key]
            group = j * res.groups // nodes
            assert c.status != RECON
            assert c.status == TAIL or (c.status == LIVE and c.last_group < group)
        assert len(set(res.placed.values())) == len(res.placed)
        if groups == 1:  # one batched GEMM: only the unused tail can take products
            assert len(res.placed) == min(res.count(TAIL), 7 ** R)


@pytest.mark.parametrize("N,R", [(4096, 1), (8192, 2), (8192, 3), (16384, 4), (32768, 4)])
def test_leaf_operands_cover_a_b_s(N, R):
    """Every leaf block of A, B and of the S areas of all nodes is a direct operand of exactly one
    leaf multiplication; only the unused tail of S is read by none; 2 x 7^R operand reads."""
    from gemm_l2.layout import layout
    from gemm_l2.reuse import TAIL, analyse

    res = analyse(layout(N, R), 1)
    for key, c in res.cells.items():
        if c.status == TAIL:
            assert c.readers == 0, key
        else:
            assert c.readers == 1, key
            assert c.side in ("L", "R") and c.reader.startswith("P")
    assert sum(c.readers for c in res.cells.values()) == 2 * 7 ** R
    assert sum(1 for k, c in res.cells.items() if k[0] in "AB" and c.readers == 1) == 2 * 4 ** R


@pytest.mark.parametrize("R", [1, 2, 3, 4, 5])
def test_early_products_stages(R):
    """Products that can start after the encoder / after sums level s: 3^R, 4 * 7^(s-1) * 3^(R-s)."""
    from gemm_l2.overlap import expected_counts, products, stage_counts, timeline

    assert stage_counts(R) == expected_counts(R)
    assert sum(expected_counts(R)) == 7 ** R
    # stage 0: only m1, m2, m4 on every level
    for p in products(R):
        assert (p.stage == 0) == all(c in (0, 1, 3) for c in p.path)
    tl = timeline(16384, R, 300, 900)
    assert tl.overlapped_ms > 0 and tl.serial_ms > 0
