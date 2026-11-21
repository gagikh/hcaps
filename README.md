# CAPS on Hopper: Strassen–Winograd in the alternative basis on the NVIDIA H100

Fast matrix multiplication of half-precision matrices (f16 data, f16 or f32 accumulation) on the
NVIDIA H100: the Strassen–Winograd recursion in the alternative basis of Karstadt and Schwartz,
with a leaf GEMM of our own written for Hopper (TMA, mbarrier pipeline, `wgmma`).

Two schedules of the same recursion:

- **BFSCaps** (breadth-first): encode `A` and `B` in place, form the sums of all levels, run all
  `7^R` leaf products as one batched GEMM, reconstruct, decode. Fast, but needs `3.19×` the memory
  of a plain GEMM at `R = 2`.
- **DFSCaps** (depth-first top level): for each of the 7 root products, form its operands and the
  sums of its sub-tree, then run its `7^(R-1)` leaves as one batched GEMM whose epilogue adds every
  product, with its sign, straight into the blocks of `C` (TMA bulk reduce-add). No product buffers
  and no reconstruction pass: `1.29×` the memory of a plain GEMM at `R = 2`, at comparable speed for `N ≥ 32768`. The
  whole run is one CUDA graph; an overlapped mode forms the next product's operands on a second
  stream.

A paper describing the algorithm, the memory layout, the kernels and the measurements is in
progress and will be linked here.

## Results on the H100 (80 GB HBM3)

Time in ms, `cuBLAS best` = the fastest cuBLASLt heuristic candidate for each size, chosen by
timing; run-to-run variation about ±3 %. Speed-up = time of cuBLAS best / our time. All numbers
are from the CSV files in [`data/h100/`](data/h100/). Memory is measured with `cudaMemGetInfo`
while each method runs (cuBLAS: `A`, `B`, `C` and its workspace).

Both schedules work in place: they overwrite `A` and `B` with their alternative-basis encoding,
and BFSCaps returns `C` in its own buffer (`get_result()`). A caller that still needs `A` and `B`
must keep a copy; cuBLAS leaves them unchanged.

| N | cuBLAS best (f16 acc.) | ours, best schedule | speed-up | memory, DFSCaps R=2 vs cuBLAS best |
|---|---|---|---|---|
| 16384 | 10.88 | 10.11 (BFS, R=1) | 1.08× | 2.1 vs 1.7 GB |
| 32768 | 87.55 | 76.87 (BFS, R=1) | 1.14× | 8.3 vs 6.5 GB |
| 65536 | 682.2 | 566.3 (DFS, R=2) | 1.20× | 33.3 vs 25.8 GB |

Schedule = traversal of the recursion tree: BFS (breadth-first, BFSCaps), DFS (depth-first at the
root, DFSCaps) and overlapped DFS (the sums of the next root product run during the current leaf GEMM).

### Repeatability

Five independent runs per accumulation (separate jobs, each with its own warm-up and cuBLAS
candidate selection); time relative to cuBLAS best of the same run, median [smallest–largest]
([`data/h100/repeats/`](data/h100/repeats/); `rep_*` f16, `rep32_*` f32 accumulation):

| acc. | N | cuBLAS best, ms (range) | our GEMM | BFSCaps R=1 | BFSCaps R=2 | DFSCaps R=2 |
|---|---|---|---|---|---|---|
| f16 | 16384 | 10.97 (10.76–11.02) | 0.923 [0.920–0.943] | 0.919 [0.917–0.939] | 1.107 [1.105–1.124] | 1.195 [1.189–1.217] |
| f16 | 32768 | 87.55 (87.55–87.55) | 0.913 [0.913–0.921] | 0.873 [0.872–0.874] | 0.898 [0.897–0.899] | 0.911 [0.911–0.911] |
| f16 | 65536 | 687.59 (682.32–688.08) | 0.931 [0.930–0.938] | 0.859 [0.858–0.866] | 0.824 [0.822–0.829] | 0.821 [0.821–0.827] |
| f32 | 16384 | 10.57 (10.42–10.69) | 1.001 [0.997–1.015] | 0.993 [0.981–1.001] | 1.151 [1.132–1.167] | 1.245 [1.235–1.260] |
| f32 | 32768 | 83.29 (82.51–84.09) | 1.005 [1.003–1.006] | 0.957 [0.951–0.963] | 0.978 [0.966–0.989] | 0.971 [0.963–0.979] |
| f32 | 65536 | 669.05 (662.68–669.33) | 1.003 [1.002–1.003] | 0.917 [0.915–0.924] | 0.875 [0.874–0.884] | 0.873 [0.872–0.881] |

### Sizes that are not powers of two

Both schedules accept any `N = k · 2^R` with the leaf size `k` a multiple of 256, e.g.
`N = 3 · 2^14 = 49152`. DFSCaps runs every such `(N, R)`; BFSCaps is compiled per size and runs
the sizes listed in `gen/tuning_table.hpp` (12288, 24576 and 49152 besides the powers of two; add
a line for another size). Measured on the H100 (f16 accumulation, ms):

| N | cuBLAS best | ours, best schedule | speed-up | memory, DFSCaps R=2 vs cuBLAS best |
|---|---|---|---|---|
| 12288 = 3·2^12 | 4.43 | 4.27 (our single GEMM) | 1.04× | 1.2 vs 1.0 GB |
| 24576 = 3·2^13 | 35.74 | 32.88 (BFS, R=1) | 1.09× | 4.7 vs 3.7 GB |
| 49152 = 3·2^14 | 283.2 | 248.3 (DFS, R=2) | 1.14× | 18.7 vs 14.6 GB |

### Comparison with SubCuber (PLDI 2026)

[SubCuber](https://github.com/microsoft/subcuber) (A. Jangda, "Compiling Strassen-like Matrix
Multiplication Algorithms to Fast CUDA Kernels", PACMPL 10(PLDI), 2026) compiles
Strassen-like algorithms into fused CUDA kernels. It supports f16 data with f32 accumulation only,
so the comparison uses our f32-accumulation run. Both were measured on the same H100 model, with
SubCuber's fastest registered kernel per size and level. Levels 0 and 2 at N = 16384 and 32768
were rerun with SubCuber built with CUDA 13.1 (the toolkit of our binaries), about 1 s of warm-up
calls and 50 timed calls, as for our code
([log](data/h100/subcuber/warm_cuda131_h100.log)); the other entries come from a CUDA 12.8 build
with 3 warm-up and 10 timed calls.

| N | cuBLAS best (f32 acc.) | ours (f32 acc.) | SubCuber level 2 | ours / SubCuber | memory: ours (DFS R=2) / SubCuber level 2 |
|---|---|---|---|---|---|
| 16384 | 10.59 | 10.48 | **10.10** | 1.04 | 2.1 / 7.3 GB |
| 32768 | 83.29 | **79.75** | 87.21 | **0.91** | 8.3 / 27.2 GB |
| 65536 | 662.9 | **584.0** | out of memory | — | 33.3 / ~107 GB (model) |

On the H100, **we beat SubCuber from N = 32768 on**: 9 % less time at 32768, and at 65536 SubCuber
cannot run at all. Its level-2 workspace (four presums of `A`, four of `B` and seven postsums per
first-level product) needs `4.2`–`4.5×` the memory of a plain GEMM, ~107 GB at 65536, and it fails
with out of memory in `cudaMalloc`
([compute-sanitizer log](data/h100/subcuber/oom_65536_level2_h100.log)), while DFSCaps needs 33.3 GB and is 1.14× faster than cuBLAS best
with f32 accumulation (1.20× with f16). **SubCuber is faster at N = 16384** (4 % less time), where its
fused kernels avoid the memory-bound passes that dominate our pipeline at medium sizes.
SubCuber's level 1 is slower than its own plain GEMM at every size we measured (up to 1.68× at
N = 65536), and most of its level-1 kernels reject the problem or crash on the H100, so level 2
is its best variant. SubCuber's
measured times and peak memory are in [`data/h100/subcuber/`](data/h100/subcuber/).

## Requirements

- An NVIDIA Hopper GPU (H100 / H200, compute capability 9.0) for the `wgmma` leaf GEMM and
  DFSCaps; other GPUs can build with `LEAF_GEMM=cublas` (BFSCaps only).
- CUDA 12.8 or later (measured with CUDA 13.1), CMake ≥ 3.17, a C++17 host compiler.
- OpenCV with the CUDA modules `core`, `cudaarithm`, `cudev`.
- Qt 5 (`Core`, `Test`) for the command-line parser and the unit tests.
- Intel oneMKL (CPU reference GEMM in the tests).
- Python 3 with `py/requirements.txt` for the plots.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=90a-real -DLEAF_GEMM=wgmma ..
make -j
```

Main CMake options: `LEAF_GEMM` (`wgmma` or `cublas`), `CUDA_ARCH` (`90a-real` for H100/H200),
`LOOPS_COUNT_AS_TEMPLATE_ARGS` (ON: the tuned loop counts of `gen/tuning_table.hpp` are compiled
in; OFF: needed by the tuner), `L2_CACHE_POLICY`, `PRODUCTION_CODE`.

## Run

```bash
./bin/test_unit_tests                       # BFSCaps, DFSCaps and the leaf GEMM against cuBLAS
./bin/test_gemm_benchmark --accumulator=f16 --max-dim=131072 --results=run_f16
./bin/test_gemm_benchmark --accumulator=f32 --max-dim=131072 --results=run_f32
./bin/test_gemm_benchmark --precision=1 --results=run_precision
./bin/test_gemm_benchmark --sizes=12288,24576,49152 --results=run_kx   # any N = k * 2^R
./bin/test_diagnostics dfs-check 65536 2    # DFSCaps modes against BFSCaps, time and memory
```

The benchmark writes `relative_costs.csv` (time per method and N), `batched_costs.csv` (the leaf
GEMMs alone) and `memory.csv` (device memory per method and N). `python py/plots.py --data <dir>`
draws them.

## Layout

| Folder | Contents |
|---|---|
| `include/`, `src/` | BFSCaps (`bfs_caps.*`), DFSCaps (`dfs_caps.*`), the wgmma leaf GEMM (`wgmma_gemm.*`), the element-wise passes (`*.cuh`) |
| `tst/` | unit tests, benchmark, diagnostics, precision |
| `gen/` | tuned loop counts per GPU (written by `test_diagnostics tune`) |
| `py/` | plots, the L2 / traffic model of the passes, viewers |
| [`data/h100/`](data/h100/) | the measurements in the tables above |

## License

MIT, see `LICENSE`.
