# Python tools

Everything that does not run on the GPU: the L2 simulation of the pipeline, the viewer, the code generator, the benchmark plots and the GPU-free diagnostics. They used to be
C++ (`src/L2.cpp`, `src/mult_order.cpp`, `experimental/schedule_viz`, `tst/generator.cpp`,
`schwartz::tree_node_t`, QCustomPlot with `tst/graph_plot` / `tst/plotting_utils`, part of
`tst/diagnostics.cpp`); the GEMM library no longer depends on any of it.

```
pip install -r py/requirements.txt
python py/viewer.py               # interactive viewer
python py/generate.py             # images and graphs into docs/ (--cuh: gen/depth_<R>_functions.cuh)
python py/plots.py                # graphs of ./bin/test_gemm_benchmark results (plots/data -> plots/)
python py/plots.py --store        # keep the last run: <gpu>/<run>/data + its graphs (e.g. 4090/2026-09-26_14-05-33)
python py/diagnostics.py l2-warps [N R B] | select-nodes [N R]
python -m pytest py/tests         # tests
```

The "Dependency graph" tab and the `.svg` files need Graphviz: install it from
https://graphviz.org/download/ and put its `bin` folder on PATH, or set `GRAPHVIZ_DOT` to `dot.exe`.

| module | content |
|---|---|
| `gemm_l2/tree.py` | Strassen-Winograd tree at leaf-block granularity: `biparted(R)` (leaf multiplication -> input blocks, every sum formed on the fly), `coefficients(R)` |
| `gemm_l2/l2_sim.py` | DRAM traffic of the whole pipeline (encode, forward, mma leaf or fused kernel, reconstruct, decode) through an LRU of the L2 size, from the row loops of the passes and the tile order of the GEMM stages; the fused configurations are read from `src/fused_strassen.cu`, loop counts and leaf configuration from `gen/tuning_table.hpp` |
| `gemm_l2/images.py` | parameter validation, title and legend bands of the images |
| `gemm_l2/deps.py` | read/write dependency graph of the BFS pipeline below a start level (real buffers: quadrant views, S1..S6, products, C written into S1..S4), DOT output, write-back LRU traffic of grouped execution |
| `gemm_l2/layout.py` | device memory layout of the BFS buffers: port of `BFS::initialize()` / `create_intermediate_sums_layout()` (S ROIs of every node, leaf product buffers) |
| `gemm_l2/reuse.py` | where can the leaf products go? A, B and S split into leaf x leaf blocks: which leaf multiplication reads each block directly (every block exactly one), whether the leaf reconstruction writes it; greedy placement of the products for 1, 7, 49, ... GEMM groups and the memory saved |
| `layout_view.py` | the "Memory layout" tab (QGraphicsScene) |
| `gemm_l2/overlap.py` | which leaf products can start before all sums exist: stage of every product (0 = right after the encoder, s = after sums level s; 3^R and 4 * 7^(s-1) * 3^(R-s) products), model timeline serial (BFS) vs overlapped (GraphCAPS) |
| `overlap_view.py` | the "Early products" tab |
| `gemm_l2/tile_order.py` | tile order of the mma leaf GEMM (port of the block -> tile mapping of `src/mma_gemm.cu`): launch order, waves, panels per wave, images |
| `gemm_l2/codegen.py` | leaf products as expression trees: `gen/depth_<R>_functions.cuh` and `docs/test.dot` |
| `plots.py` | benchmark graphs from the CSV files of `tst/results.hpp` (matplotlib) |
| `diagnostics.py` | `l2-warps`, `select-nodes` (formerly `test_diagnostics`) |
| `gemm_l2/graphviz.py` | rendering with the Graphviz engines (dot, neato, fdp, sfdp, circo, twopi, osage, patchwork) |

## Viewer tabs

- **L2 simulation**: `l2_sim.py` for N, R, the L2 size and the GPU preset (SMs, DRAM bandwidth):
  DRAM read / write bytes per stage against the minimum (every input read once, every result
  written once) and the time at the DRAM bandwidth, then the GEMM stage for every tile order.
  Leaf: the mma leaf GEMM (config and order) or the fused last level; the loop counts and the mma
  configuration come from `gen/tuning_table.hpp` ("Tuned") and can be changed.
- **Dependency graph**: `deps.py` graph of the whole tree for R (forward levels, the 7^R leaf GEMMs,
  reconstruct; R <= 3), rendered with the selected Graphviz engine. Wheel zooms.
- **Encoder**: the alternative basis transforms (`gemm_l2/encode.py`): encode A, encode B or decode C
  as leaf-block grids, level by level (blocks read with + / -, written, changed earlier), what every
  block holds in terms of the original blocks (hover; written into the blocks for R <= 2), the rows
  of one row loop, and the bytes per level and per loop against L2 with the DRAM traffic of
  `l2_sim.py`.
- **L2 traffic**: DRAM traffic per subtree (of "subtree levels" levels) when `group` subtrees run forward -> GEMM -> reconstruct
  together, against today's whole-level stages; with ring scratch buffers and discard.
- **Memory layout**: A, B, the S buffer and the leaf product buffers to scale (one unit = one half),
  as `BFS::initialize()` places them. S is one address range, folded into columns; every node's
  S1..S6 ROI is coloured by level, the dashed frame marks the first L2-size bytes. Hover for
  address ranges, click a node to highlight its parent and children. Save as SVG or PNG.
  *Mark blocks for products* / *Products in A, B, S* (with *GEMM groups*) colour every leaf x leaf block of A, B and S:
  green = can take products (from the group after its last reader), red = level R-1 S1..S4 (the
  leaf reconstruction writes there), dark = read by the last group; *Products in A, B, S* places the
  products there and shows the memory saved, to decide whether reusing S is worth it.
- **Early products**: A, B and the sums S1..S6 of every node (level by level, in leaf blocks) coloured by when
  they exist, next to the 7^R leaf products (one row m1..m7 per level R-1 node) coloured by the stage
  they can start at: stage 0 needs only the encoded A and B (quadrant products, m1/m2/m4 on every level),
  stage s the sums of level s. Below, a model timeline of the serial BFS schedule (all sums, then all
  leaf GEMMs) and the overlapped GraphCAPS one (leaf GEMM group s starts after sums level s), from the
  leaf GEMM TFLOPS and the DRAM bandwidth; optimistic (the streams are assumed not to slow each other).
  Click a product to frame the two blocks it reads, or a block to frame the product that reads it.
- **Tile order**: one leaf product of the mma leaf GEMM (leaf = N >> R, ld = N) for a
  configuration and tile order (CUTLASS swizzle, CUTLASS columns, groups of g rows): the launch
  number of every tile, coloured by wave (SMs blocks at a time), and the A row / B column panels
  the first wave reads, with its bytes per k slice and the address span of its A rows.

The L2 scheduler (`cache.py`, `schedule.py`, `mult_order.py`, the "L2 schedule" and
"Dependencies" tabs, their images and tests) is not used by the GPU code any more and moved to
`archive/cleanup_2026-10/py`.

## Difference to the old C++ schedules (archived scheduler)

`scheduleBatches_v2` in `src/L2.cpp` copied its `LRUCache` (the forward pass's final cache is the
start of the backward pass) with the implicit copy constructor. The copy kept iterators into the
original `std::list`, so the backward pass worked on corrupted state (undefined behaviour). The
Python port is exact otherwise: it reproduces the C++ schedules once that copy is fixed
(`tests/data/schedule_v2_cpp.json`), and the tree-order loads of every old order file. The old
`gen/mult_order` files came from the buggy version, so regenerated orders differ where the backward
pass won.
