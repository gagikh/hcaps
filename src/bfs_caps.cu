/**
 * @file bfs_caps.cu
 * @brief The cooperative CAPS kernel (bfs_kernel) and its host side: kernel selection, launch
 *        geometry, launch (BFSCaps::process) and loop-count tuning (BFSCaps::tune)
 * @author GEMM Project
 * @date 2025
 *
 * CAPS here = Strassen-Winograd in an alternative basis, evaluated breadth-first over the
 * recursion tree. Level 0 is the root (the whole N x N product); every node of level l has 7
 * children (MULTS_COUNT) of half its size; the 7^R leaves ((N >> R) square) are multiplied by a
 * batched GEMM (BFSCaps::batched_leaf(): cuBLAS or our mma.sync kernel) that is NOT in this file.
 * One pipeline run is
 *
 *   ENCODE                          A, B -> alternative basis, in place (alternative_basis.cuh)
 *   BILINEAR                        level by level, the sums S1..S6 of every node (bilinear.cuh)
 *   ---- leaf GEMM (separate kernel) ----
 *   RECONSTRUCT_LEAF_MULTS          7 leaf products -> product of each level R - 1 node
 *   RECONSTRUCT_INTERMEDIATE_MULTS  children's products -> parent's product, up to the root
 *                                   (both in reconstruct.cuh)
 *   DECODE                          root product (= C) back from the alternative basis, in place
 *
 * All phases live in one template kernel, bfs_kernel<operation, N, R, ...>; the bit mask
 * `operation` (BFSCaps::ENCODE | BFSCaps::BILINEAR | ..., include/bfs_caps.hpp) selects the phases at compile
 * time. It is launched cooperatively with a persistent grid that walks "virtual blocks" in a
 * grid-stride loop and separates dependent levels with grid.sync().
 *
 * Conventions shared by the device functions below:
 *  - T is half8; a thread moves DIM_SCALE = sizeof(T) / sizeof(half)
 *    halves per access. x coordinates and leading dimensions (global_lds = N / DIM_SCALE) are in
 *    units of T, y coordinates in rows.
 *  - A virtual block is a WARP_SIZE x WARPS_IN_Y_DIR (32 x 4) thread tile. Virtual block id ->
 *    tile_x = id % tiles_count_x, tile_y = id / tiles_count_x (opt_mod / opt_div: powers of two
 *    only). The nodes of one level are stacked vertically in the virtual y space:
 *    node_idx = yy / rows_per_node. A thread whose node_idx is past the last node leaves the
 *    grid-stride loop (ids only grow), but still reaches the following grid.sync().
 *  - Row loops: with `loops` row loops, loop `row` handles rows row, row + loops, row + 2 * loops,
 *    ... of every node region on every level. Every region height is a multiple of loops, so the
 *    quadrant partners at distance M (A11 / A21, S1 / S3, ...) fall into the same loop, and one
 *    loop is a closed chain of level-to-level dependencies over 1 / loops of the data: small
 *    enough to stay in L2 from one level to the next. loops = 1: all rows at once.
 *  - Nodes are addressed through constant memory: __intermediate_address_cache__ holds the
 *    internal nodes (levels 0 .. R - 1) in breadth-first order, level after level (the 7 children
 *    of node i of a level are at start(next level) + 7 * i); __mult_address_cache__ the leaf
 *    product buffer of each level R - 1 node.
 *  - "complexity = n" / "n mem ops" comments are hand estimates of the per-thread instruction
 *    count / memory accesses of a step, not measurements.
 *
 * bfs_caps.ini, included at the end, holds the explicit instantiations of BFSCaps::process<operation>.
 */

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>
#include <core.hpp>

#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>

#include "bfs_caps.hpp"
#include <tuning.hpp>

#include "alternative_basis.cuh"
#include "bilinear.cuh"
#include "reconstruct.cuh"
#include "bfs_caps.cuh"
#include <common.hpp>
#include <perf.hpp>

#include <limits>
#include <string>
#include <utility>

// Node address tables, filled by BFSCaps::store_address_cache() (from initialize()). Constant memory
// fits the access pattern: all lanes of a warp share threadIdx.y and therefore the node, so a
// table read is a uniform (broadcast) access.
// 3 float pointers in each, 3 * 8 = 24 bytes
/** @brief Internal nodes of levels 0 .. R - 1 in breadth-first order: operands m_L, m_R and the
 *         node's S area m_S (sums in the forward pass, the node's product in the backward pass) */
__constant__ BFSCaps::layer_info_t __intermediate_address_cache__[ADDRESS_CACHE_SIZE];
/** @brief Per node of level R - 1 (same order): its 7 leaf products, back to back, each
 *         (N >> R) x (N >> R) halves with leading dimension N >> R (written by the leaf GEMM) */
__constant__ float *__mult_address_cache__[MULT_CACHE_SIZE];

/**
 * @brief Store address cache in GPU constant memory
 * @param intermediate_cache Vector of intermediate layer information
 * @param mult_cache Vector of multiplication addresses
 * @param stream CUDA stream for asynchronous operations
 * @return true if cache was stored successfully, false otherwise
 * 
 * Stores the address cache in GPU constant memory for fast access during
 * kernel execution. This optimizes memory access patterns and reduces
 * global memory traffic.
 *
 * The copies are ordered on `stream`, so kernels launched later on the same stream see them.
 * Entry order must match what the kernels expect: breadth-first, level by level (see the file
 * header).
 */
bool BFSCaps::store_address_cache(const std::vector<BFSCaps::layer_info_t> &intermediate_cache,
                              const std::vector<float *> &mult_cache,
                              cv::cuda::Stream &stream)
{
#if REPORT_STATISTICS
    std::cerr << "Address cache size: " << sizeof(BFSCaps::layer_info_t) << std::endl;
#endif

    // Validate both caches before touching constant memory, so a failure never leaves
    // one table updated and the other stale.
    if (intermediate_cache.empty() || intermediate_cache.size() > ADDRESS_CACHE_SIZE) {
        std::cerr << "Intermediate address cache has " << intermediate_cache.size()
                  << " entries, constant memory holds 1.." << ADDRESS_CACHE_SIZE << std::endl;
        return false;
    }
    if (mult_cache.empty() || mult_cache.size() > MULT_CACHE_SIZE) {
        std::cerr << "Multiplication address cache has " << mult_cache.size()
                  << " entries, constant memory holds 1.." << MULT_CACHE_SIZE << std::endl;
        return false;
    }

    // Pageable host source: the copy is staged before the call returns, so the vectors may be
    // released afterwards. Errors (including earlier asynchronous ones) are reported here.
    const auto cuda_stream = (cudaStream_t) stream.cudaPtr();
    CUDA_CHECK(cudaMemcpyToSymbolAsync(__intermediate_address_cache__,
                                       intermediate_cache.data(),
                                       intermediate_cache.size() * sizeof(intermediate_cache[0]),
                                       0, // offset in constant memory
                                       cudaMemcpyHostToDevice,
                                       cuda_stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(__mult_address_cache__,
                                       mult_cache.data(),
                                       mult_cache.size() * sizeof(mult_cache[0]),
                                       0, // offset in constant memory
                                       cudaMemcpyHostToDevice,
                                       cuda_stream));
    return true;
}

/**
The Strassen-Winograd variant (alternative basis) applied at every node, for reference.

Forward sums (bilinear.cuh), stored in the node's S area (layout: BFSCaps::sums_rows()):
• S1 = A21 + A22
• S2 = A22 - A12
• S3 = A22 - A11
• S4 = B22 - B11
• S5 = B21 + B22
• S6 = B22 - B12

Products = the 7 children ("no wait": both operands are quadrants of the node's own operands;
"wait for": a sum of the node is needed first):
• m1 = A11 * B11 -> no wait
• m2 = A12 * B21 -> no wait
• m3 = A21 * S4  -> wait for S4
• m4 = A22 * B22 -> no wait
• m5 = S1 * S5   -> wait for S1 and S5
• m6 = S2 * S6   -> wait for S2 and S6
• m7 = S3 * B12  -> wait for S3

Reconstruction (reconstruct.cuh), into the node's S area:
• C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6, C22 = m5 + m6 - m2 - m4
*/

/**
 * Forward-pass sums (BFSCaps::BILINEAR): for the tree levels 0 .. levels - 1, the six sums S1..S6 of
 * every node (bilinear<>::process(), bilinear.cuh) from the quadrants of its operands, written
 * into the node's S area, where they become operands of the node's children.
 *
 * levels: the sums of the nodes of levels 0 .. levels - 1 (BFSCaps::BILINEAR passes R: all).
 *
 * Loop structure: for row in [0, loops) { for l in 1 .. levels { grid-stride over the virtual
 * blocks; grid.sync() } }. Iteration l handles the 7^(l - 1) nodes of tree level l - 1, whose
 * quadrants are M = N >> l square; each node contributes RR = M / loops quadrant rows (rows row,
 * row + loops, ...), each M / DIM_SCALE T wide. One thread forms all six sums of one T-sized
 * element position of the quadrant.
 *
 * @tparam N               matrix dimension (halves)
 * @tparam R               recursions
 * @tparam levels          number of tree levels whose sums are formed (R or R - 1)
 * @tparam T               vector type of the accesses (half8)
 * @tparam bilinear_loops  row loop count (template argument with LOOPS_COUNT_AS_TEMPLATE_ARGS,
 *                         else a run-time argument); a power of two <= N >> R
 * @tparam cache           L2-favouring load/store variants (fetch_with_policy(), utils.cuh)
 * @tparam calculate_A     form S1..S3 (A side)
 * @tparam calculate_B     form S4..S6 (B side)
 * @param barrier          grid group of the cooperative launch, synced after every level
 * Precondition: the operands of level 0 (encoded A and B) are complete.
 */
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
// top-level loop = N / rows
template<unsigned int N,
         unsigned int R,
         unsigned int levels,
         class T,
         unsigned int bilinear_loops,
         bool cache,
         bool calculate_A,
         bool calculate_B>
__device__ __forceinline__ void bilinear_kernel(cg::grid_group &barrier)
#else
template<unsigned int N,
         unsigned int R,
         unsigned int levels,
         class T,
         bool cache,
         bool calculate_A,
         bool calculate_B>
__device__ __forceinline__ void bilinear_kernel(cg::grid_group &barrier, unsigned int bilinear_loops)
#endif
{
    constexpr auto DIM_SCALE = sizeof(T) / sizeof(half);
    constexpr auto dim_x = WARP_SIZE;
    constexpr auto dim_y = BFSCaps::WARPS_IN_Y_DIR;
    constexpr auto global_lds = N / DIM_SCALE; // ld is in half2 or half8

    // rows of the full N-row matrix per loop; a node of level l - 1 gets rows >> l of them
    const auto rows = N / bilinear_loops;

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int total_blocks_count = gridDim.y;

    // each loop outputs "rows" rows (scaled down per level, see RR)
    // accessing 3*N/2 * (R / 2) halves
GEMM_ROW_LOOP_UNROLL
    for (auto row = 0; row < bilinear_loops; ++row) {
        // first index (in __intermediate_address_cache__) and count of the nodes of the current
        // tree level; level 0 is the root alone
        auto m_nodes_start = 0;
        auto m_nodes_count = 1;

#pragma unroll
        for (auto l = 1; l <= levels; ++l) {
            const auto M = N >> l;      // quadrant size of the level l - 1 nodes
            const auto RR = rows >> l;  // = M / bilinear_loops: rows per node in this loop
            // virtual blocks across one quadrant row (a block row is 32 T = 32 * DIM_SCALE halves)
            const auto tiles_count_x = M / (dim_x * DIM_SCALE);

            // grid-stride walk over the virtual blocks of this level
            unsigned int block_start_id = blockIdx.y;
            do {
                // calculate virtual pos = 18
                // Calculate offsets = 42
                const auto id = block_start_id;

                // tile id in x dir, complexity = 2
                const auto tile_x = opt_mod<pow2_v<N>>(id, tiles_count_x);
                const auto xx = tile_x * dim_x + x;

                // tile id in y dir, complexity = 2
                const auto tile_y = opt_div<pow2_v<N>>(id, tiles_count_x);
                const auto yy = tile_y * dim_y + y;

                // the RR-row slices of the nodes are stacked in yy
                const auto node_idx = opt_div<pow2_v<N>>(yy, RR);
                // 1 check: past the last node of the level (later ids are too)
                if (node_idx >= m_nodes_count) {
                    break;
                }
                const auto xxx = xx;

                // complexity = 2
                // row inside the quadrant: the k-th row of this loop is row + k * bilinear_loops
                const auto yyy = row + opt_mod<pow2_v<N>>(yy, RR) * bilinear_loops;
                const auto m_node = m_nodes_start + node_idx;

                // global_lds and xxx are in units of T
                // complexity = 42,  memory ops = 11
                // last level: streaming (last use) loads/stores when LAST_USE_HINTS is on
                bilinear<cache, calculate_A, calculate_B>::process(m_node,
                                                                      global_lds,
                                                                      M,
                                                                      xxx,
                                                                      yyy,
                                                                      LAST_USE_HINTS && l == levels);
                // complexity 1
                block_start_id += total_blocks_count;

            } while (true);
            // the next level reads the sums just written (by other blocks)
            barrier.sync();

            // next tree level: 7x as many nodes, stored right after this level
            m_nodes_start = m_nodes_start + m_nodes_count;
            m_nodes_count = m_nodes_count * MULTS_COUNT;
        }
    }
}

/**
 * Backward pass, first step (BFSCaps::RECONSTRUCT_LEAF_MULTS), for one row loop `row`: every node of
 * level R - 1 combines its 7 leaf products (the buffer __mult_address_cache__[node], written by the
 * leaf GEMM) into its own product C11..C22, stored into the node's S area
 * (block_reconstruct<T>::leaf_block_reconstruct(), reconstruct.cuh). Ends with grid.sync().
 *
 * M = N >> R is the leaf size (= a quadrant of a level R - 1 node); each node contributes
 * rows = M / reconstruct_loops rows (row, row + loops, ...), each M / DIM_SCALE T wide.
 *
 * @param barrier            grid group of the cooperative launch
 * @param reconstruct_loops  row loop count, a power of two <= M (template argument with
 *                           LOOPS_COUNT_AS_TEMPLATE_ARGS)
 * @param row                current row loop, 0 .. reconstruct_loops - 1 (looped in bfs_kernel)
 * @param m_nodes_start      index of the first level R - 1 node in __intermediate_address_cache__
 * @param m_nodes_count      number of level R - 1 nodes (7^(R - 1))
 */
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
template<unsigned int N, unsigned int R, class T, bool cache, unsigned int reconstruct_loops>
__device__ __forceinline__ void reconstruct_leaf_mults_kernel(cg::grid_group &barrier,
                                                              unsigned int row,
                                                              unsigned int m_nodes_start,
                                                              unsigned int m_nodes_count)
#else
template<unsigned int N, unsigned int R, class T, bool cache>
__device__ __forceinline__ void reconstruct_leaf_mults_kernel(cg::grid_group &barrier,
                                                              unsigned int reconstruct_loops,
                                                              unsigned int row,
                                                              unsigned int m_nodes_start,
                                                              unsigned int m_nodes_count)
#endif
{
    constexpr auto dim_x = WARP_SIZE;
    constexpr auto dim_y = BFSCaps::WARPS_IN_Y_DIR;
    constexpr auto last_layer = R;

    constexpr auto M = N >> last_layer; // leaf size
    constexpr auto DIM_SCALE = sizeof(T) / sizeof(half);

    // virtual blocks across one leaf row (x is in units of T)
    constexpr auto tiles_count_x = M / (DIM_SCALE * dim_x);
    //const auto tiles_count_y = M / dim_y;

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int total_blocks_count = gridDim.y;

    constexpr auto global_lds = N / DIM_SCALE; // ld is in half2 or half8
    const auto rows = M / reconstruct_loops; // rows per node in this loop

    // grid-stride walk over the virtual blocks
    unsigned int block_start_id = blockIdx.y;
    do {
        /**
         * calculate virtual pos: complexity = 21
         * offsets
         */
        const auto id = block_start_id;
        // complexity = 2
        const auto tile_x = opt_mod<pow2_v<N>>(id, tiles_count_x);
        const auto xx = tile_x * dim_x + x;

        // complexity = 5
        const auto tile_y = opt_div<pow2_v<N>>(id, tiles_count_x);
        const auto yy = tile_y * dim_y + y;

        // the row slices of the nodes are stacked in yy
        const auto node_idx = opt_div<pow2_v<N>>(yy, rows);

        // complexity = 1: past the last node (later ids are too)
        if (node_idx >= m_nodes_count) {
            break;
        }
        const auto xxx = xx;
        // complexity = 2
        const auto yyy = row + opt_mod<pow2_v<N>>(yy, rows) * reconstruct_loops;

        // complexity = 2
        const auto s_node = m_nodes_start + node_idx; // the node's S area (receives its product)
        const auto m_node = node_idx; // its leaf products (__mult_address_cache__ holds level R - 1 only)

        // 13 mem ops, 41 ops
        block_reconstruct<T, cache>::leaf_block_reconstruct(s_node, m_node, global_lds, M, xxx, yyy);
        // complexity = 1
        block_start_id += total_blocks_count;
    } while (true);
    // the intermediate reconstruction (or DECODE) reads the products written here
    barrier.sync();
}

/**
 * Backward pass, remaining levels (BFSCaps::RECONSTRUCT_INTERMEDIATE_MULTS), for one row loop `row`:
 * for l = R - 1 down to 1, every node of level l - 1 forms its product from the products of its
 * 7 children (level l, already in their S areas) with
 * block_reconstruct<T>::intermediate_block_reconstruct(), into its own S area; grid.sync() after
 * every level. Afterwards the root's S area holds C, still in the alternative basis (DECODE).
 * Needs R > 1 (with R = 1 the level R - 1 node is the root).
 *
 * M = N >> l is the size of a child (= a quadrant of the parent); rows = M / reconstruct_loops.
 *
 * @param barrier            grid group of the cooperative launch
 * @param reconstruct_loops  row loop count (template argument with LOOPS_COUNT_AS_TEMPLATE_ARGS)
 * @param row                current row loop, 0 .. reconstruct_loops - 1
 * @param m_nodes_start      first index of level R - 1 in __intermediate_address_cache__ (by value,
 *                           walked up one level per iteration)
 * @param m_nodes_count      number of level R - 1 nodes (likewise)
 */
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
template<unsigned int N, int R, class T, bool cache, unsigned int reconstruct_loops>
__device__ __forceinline__ void reconstruct_intermediate_mults_kernel(cg::grid_group &barrier,
                                                                      unsigned int row,
                                                                      int m_nodes_start,
                                                                      int m_nodes_count)
#else
template<unsigned int N, int R, class T, bool cache>
__device__ __forceinline__ void reconstruct_intermediate_mults_kernel(cg::grid_group &barrier,
                                                                      unsigned int reconstruct_loops,
                                                                      unsigned int row,
                                                                      int m_nodes_start,
                                                                      int m_nodes_count)
#endif
{
    const unsigned int total_blocks_count = gridDim.y;
    constexpr auto dim_x = WARP_SIZE;
    constexpr auto dim_y = BFSCaps::WARPS_IN_Y_DIR;
    constexpr auto DIM_SCALE = sizeof(T) / sizeof(half);

    constexpr auto global_lds = N / DIM_SCALE; // ld is in half2 or half8

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;

#pragma unroll
    for (auto l = R - 1; l > 0; --l) {
        const auto src_nodes_start = m_nodes_start; // the children (level l)
        // step up to the parents (level l - 1), stored just before level l
        m_nodes_count /= MULTS_COUNT;
        m_nodes_start -= m_nodes_count;
        const auto M = N >> l;                   // child size = quadrant of the parent
        const auto rows = M / reconstruct_loops; // rows per parent in this loop

        // virtual blocks across one quadrant row (x is in units of T)
        const auto tiles_count_x = M / (DIM_SCALE * dim_x);
        //const auto tiles_count_y = M / dim_y;

        unsigned int block_start_id = blockIdx.y;
        do {
            const auto id = block_start_id;
            const auto tile_x = opt_mod<pow2_v<N>>(id, tiles_count_x);
            const auto xx = tile_x * dim_x + x;

            const auto tile_y = opt_div<pow2_v<N>>(id, tiles_count_x);
            const auto yy = tile_y * dim_y + y;

            // tiles_count_x * (rows / dim_y) virtual blocks cover one parent's row slice
            // (with rows < dim_y one virtual block spans several parents)
            const auto node_idx = opt_div<pow2_v<N>>(yy, rows);

            if (node_idx >= m_nodes_count) {
                break;
            }

            const auto dst_node = m_nodes_start + node_idx;
            const auto xxx = xx;

            // complexity = 2
            const auto yyy = row + opt_mod<pow2_v<N>>(yy, rows) * reconstruct_loops;
            // the 7 children of node_idx are consecutive, starting at src_nodes_start + 7 * node_idx
            block_reconstruct<T, cache>::intermediate_block_reconstruct(dst_node,
                                                                        src_nodes_start
                                                                            + MULTS_COUNT * node_idx,
                                                                        global_lds,
                                                                        M,
                                                                        xxx,
                                                                        yyy);
            block_start_id += total_blocks_count;
        } while (true);
        // the next (upper) level reads the products written here
        barrier.sync();
    }
}

/**
 * The CAPS kernel. Runs the phases selected by the bit mask `operation` (include/bfs_caps.hpp), always
 * in the order ENCODE -> BILINEAR -> RECONSTRUCT_LEAF_MULTS -> RECONSTRUCT_INTERMEDIATE_MULTS ->
 * DECODE; phases not in the mask are removed at compile time. The leaf GEMM runs between BILINEAR
 * and the reconstruction in a separate kernel, so the pipeline launches FORWARD (ENCODE | BILINEAR)
 * and BACKWARD (both reconstructions | DECODE).
 *
 * Launch (BFSCaps::process(), launch_geometry()): cooperative, block = WARP_SIZE x WARPS_IN_Y_DIR
 * (128 threads, matching __launch_bounds__), grid = 1 x (co-resident blocks / divisor). Each phase
 * grid-strides over its virtual blocks with blockIdx.y / gridDim.y and separates dependent levels
 * with grid.sync(); every thread reaches every barrier (the `break`s only leave grid-stride loops).
 * Node addresses come from the constant-memory tables (BFSCaps::store_address_cache()).
 *
 * @tparam operation          OR of BFSCaps::ENCODE, BILINEAR, RECONSTRUCT_LEAF_MULTS,
 *                            RECONSTRUCT_INTERMEDIATE_MULTS, DECODE
 * @tparam N                  matrix dimension (halves), R: recursions
 * encode_loops, bilinear_loops, reconstruct_loops, decode_loops: row loop count of each phase, a
 *                            power of two <= N >> R; template arguments with
 *                            LOOPS_COUNT_AS_TEMPLATE_ARGS (from gen/tuning_table.hpp), else kernel
 *                            arguments (BFSCaps::config_t, tuner)
 * @param m_nodes_start       first index of the level R - 1 nodes (leaf_parent_nodes())
 * @param m_nodes_count       their number, 7^(R - 1)
 */
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
template<unsigned int operation,
         unsigned int N,
         unsigned int R,
         unsigned int encode_loops,
         unsigned int bilinear_loops,
         unsigned int reconstruct_loops,
         unsigned int decode_loops>
__global__ void __launch_bounds__(WARP_SIZE *BFSCaps::WARPS_IN_X_DIR *BFSCaps::WARPS_IN_Y_DIR)
    bfs_kernel(unsigned int m_nodes_start, unsigned int m_nodes_count)
#else
template<unsigned int operation, unsigned int N, unsigned int R>
__global__ void __launch_bounds__(WARP_SIZE *BFSCaps::WARPS_IN_X_DIR *BFSCaps::WARPS_IN_Y_DIR)
    bfs_kernel(unsigned int encode_loops,
               unsigned int bilinear_loops,
               unsigned int reconstruct_loops,
               unsigned int decode_loops,
               unsigned int m_nodes_start,
               unsigned int m_nodes_count)
#endif
{
    typedef half8 T;
    // T: vector width of all loads/stores (8 halves, 16 B)
    // cache: L2-favouring load/store variants (utils.cuh) for data that a later level re-reads
    constexpr auto cache = L2_CACHE_POLICY;

    // to sync across the whole grid
    cg::grid_group barrier = cg::this_grid();
    if (operation & BFSCaps::ENCODE) {
        // 2 constant cache read: the root's operands, i.e. the input matrices A and B, which are
        // transformed in place, level 1 .. R
        const auto L = __intermediate_address_cache__[0].m_L;
        const auto right = __intermediate_address_cache__[0].m_R;
        auto A = const_cast<half *>(reinterpret_cast<const half *>(L));
        auto B = const_cast<half *>(reinterpret_cast<const half *>(right));
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
        alternative_basis_transform_kernel<N, R, encoder_functor_t<cache>, encode_loops>(
            barrier, A, B);
#else
        alternative_basis_transform_kernel<N, R, encoder_functor_t<cache>>(barrier,
                                                                                     A,
                                                                                     B,
                                                                                     encode_loops);
#endif
    }

    // Notes from an unused design that kept a whole R = 2 subtree per thread in shared memory
    // (2 inputs + 6 sums + 7 * 6 child sums + 7 * 7 products = 344 values of 16 bytes):
    // 2 + 704512
    // 344 = 2 + 6 + 7 * 6 + 7 * 7
    // last layer should directly store into the memory
    // each thread will need 344 * 16 = 5504 bytes
    // one warp will take 32 * 5504
    // extern __shared__ T L1[];

    // 35 registers
    if (operation & BFSCaps::BILINEAR) {
        // R = 1: the sums go only to the leaf GEMM (another kernel); no later level of this kernel
        // re-reads them, so the L2-favouring policy is not used
        constexpr bool tmp_cache = cache && R > 1;
        constexpr unsigned int levels = R;
        // both sides (S1..S3 from A, S4..S6 from B) are formed here; the flags of bilinear<>
        // allow forming one side only (an older design formed the A side on the fly)
        constexpr bool calculate_A = true;
        constexpr bool calculate_B = true;

#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
        bilinear_kernel<N, R, levels, T, bilinear_loops, tmp_cache, calculate_A, calculate_B>(
            barrier);
#else
        bilinear_kernel<N, R, levels, T, tmp_cache, calculate_A, calculate_B>(barrier, bilinear_loops);
#endif
    }

    // the leaf products come from BFSCaps::batched_leaf() (a separate kernel)

    // 4 more registers
    // Backward pass: the leaf products m_i are complete (leaf GEMM, previous kernel). The row loop
    // encloses both reconstruction phases, so one loop's rows travel from the leaves up to the root
    // while they are still in L2.
    {
GEMM_ROW_LOOP_UNROLL
        for (auto row = 0; row < reconstruct_loops; ++row) {
            if (operation & BFSCaps::RECONSTRUCT_LEAF_MULTS) {
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
                reconstruct_leaf_mults_kernel<N, R, T, cache, reconstruct_loops>(barrier,
                                                                                 row,
                                                                                 m_nodes_start,
                                                                                 m_nodes_count);
#else
                reconstruct_leaf_mults_kernel<N, R, T, cache>(barrier,
                                                              reconstruct_loops,
                                                              row,
                                                              m_nodes_start,
                                                              m_nodes_count);
#endif
            }

            // 6 more registers
            // R = 1: the level R - 1 node is the root, there is nothing above it
            if (operation & BFSCaps::RECONSTRUCT_INTERMEDIATE_MULTS && R > 1) {
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
                reconstruct_intermediate_mults_kernel<N, R, T, cache, reconstruct_loops>(barrier,
                                                                                         row,
                                                                                         m_nodes_start,
                                                                                         m_nodes_count);
#else
                reconstruct_intermediate_mults_kernel<N, R, T, cache>(barrier,
                                                                      reconstruct_loops,
                                                                      row,
                                                                      m_nodes_start,
                                                                      m_nodes_count);
#endif
            }
        }
    }

    if (operation & BFSCaps::DECODE) {
        // the root's S area holds C in the alternative basis; decoded in place (both pointers = S)
        auto S = __intermediate_address_cache__[0].m_S;
        auto A = reinterpret_cast<half *>(S);
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
        alternative_basis_transform_kernel<N, R, decoder_functor_t<cache>, decode_loops>(barrier,
                                                                                            A,
                                                                                            A);
#else
        alternative_basis_transform_kernel<N, R, decoder_functor_t<cache>>(barrier,
                                                                              A,
                                                                              A,
                                                                              decode_loops);
#endif
    }
}

/**
 * Host: the bfs_kernel<operation, N, R, ...> instantiation for (N, R) as a void * for
 * cooperative_launch(), or nullptr if none was compiled.
 *
 * With LOOPS_COUNT_AS_TEMPLATE_ARGS the table comes from gen/tuning_table.hpp (one
 * GEMM_TUNED_BFS(gpu, N, R, encode, bilinear, reconstruct, decode) line per entry), included
 * twice: first the "default" entries are registered, then the entries of the running GPU
 * (tuning::device_name()) overwrite them. Every line is instantiated for every `operation`, so the
 * table size drives the compile time. Without it, the fixed list below with run-time loop counts.
 * The table is rebuilt on every call.
 */
template<int operation>
auto select_kernel(int N, int R)
{
    std::map<int, std::map<int, void *>> table;
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
    // gen/tuning_table.hpp (include/tuning.hpp): every entry is instantiated, the "default" ones
    // are used unless the running GPU has its own
    const auto &gpu = tuning::device_name();
#define GEMM_TUNED_BFS(g, n, r, e, b, rc, d)                                                       \
    if (std::string(g) == tuning::DEFAULT_GPU) {                                                     \
        table[n][r] = (void *) bfs_kernel<operation, n, r, e, b, rc, d>;                             \
    }
#include <tuning_table.hpp>
#undef GEMM_TUNED_BFS
#define GEMM_TUNED_BFS(g, n, r, e, b, rc, d)                                                       \
    if (gpu == g) {                                                                                  \
        table[n][r] = (void *) bfs_kernel<operation, n, r, e, b, rc, d>;                             \
    }
#include <tuning_table.hpp>
#undef GEMM_TUNED_BFS
#else
    // (N, R) pairs available in a build with run-time loop counts (the tuning build); add a line
    // for a new size. All have N >> R >= 256, see launch_geometry().
    table[512][1] = (void*)bfs_kernel<operation, 512, 1>;

    table[1024][1] = (void*)bfs_kernel<operation, 1024, 1>;
    table[1024][2] = (void*)bfs_kernel<operation, 1024, 2>;

    table[2048][1] = (void*)bfs_kernel<operation, 2048, 1>;
    table[2048][2] = (void*)bfs_kernel<operation, 2048, 2>;
    table[2048][3] = (void*)bfs_kernel<operation, 2048, 3>;

    table[4096][1] = (void*)bfs_kernel<operation, 4096, 1>;
    table[4096][2] = (void*)bfs_kernel<operation, 4096, 2>;
    table[4096][3] = (void*)bfs_kernel<operation, 4096, 3>;
    table[4096][4] = (void*)bfs_kernel<operation, 4096, 4>;

    table[8192][1] = (void*)bfs_kernel<operation, 8192, 1>;
    table[8192][2] = (void*)bfs_kernel<operation, 8192, 2>;
    table[8192][3] = (void*)bfs_kernel<operation, 8192, 3>;
    table[8192][4] = (void*)bfs_kernel<operation, 8192, 4>;

    table[16384][1] = (void*)bfs_kernel<operation, 16384, 1>;
    table[16384][2] = (void*)bfs_kernel<operation, 16384, 2>;
    table[16384][3] = (void*)bfs_kernel<operation, 16384, 3>;
    table[16384][4] = (void*)bfs_kernel<operation, 16384, 4>; // may be 32

    table[32768][1] = (void*)bfs_kernel<operation, 32768, 1>;
    table[32768][2] = (void*)bfs_kernel<operation, 32768, 2>;
    table[32768][3] = (void*)bfs_kernel<operation, 32768, 3>;
    table[32768][4] = (void*)bfs_kernel<operation, 32768, 4>; // fits the H100 (80 GB), not the 4090

    table[65536][1] = (void*)bfs_kernel<operation, 65536, 1>;
    table[65536][2] = (void*)bfs_kernel<operation, 65536, 2>;

    table[131072][1] =(void*) bfs_kernel<operation, 131072, 1>;
    // N = 3 * 2^k: not a power of two (k * 2^R with leaves k a multiple of 256)
    table[12288][1] = (void*)bfs_kernel<operation, 12288, 1>;
    table[12288][2] = (void*)bfs_kernel<operation, 12288, 2>;
    table[24576][1] = (void*)bfs_kernel<operation, 24576, 1>;
    table[24576][2] = (void*)bfs_kernel<operation, 24576, 2>;
    table[24576][3] = (void*)bfs_kernel<operation, 24576, 3>;
    table[49152][1] = (void*)bfs_kernel<operation, 49152, 1>;
    table[49152][2] = (void*)bfs_kernel<operation, 49152, 2>;
#endif

    auto it = table.find(N);
    if (it != table.end()) {
        auto jt = it->second.find(R);
        if (jt != it->second.end()) {
            return jt->second;
        }
    }
    return (void *) nullptr;
}

namespace {

// The fraction of the maximal cooperative grid that is launched is config_t::m_grid_divisor
// (default 4), the same for process() and the tuner (the tuner used to time with 3 while process()
// ran with 4, i.e. it tuned the loop counts for a different grid).

// Grid and block of a bfs_kernel launch.
struct launch_geometry_t
{
    dim3 grid;
    dim3 block;
};

// Block/grid for a cooperative bfs_kernel launch (shared by process() and the tuner).
//
// Block: one virtual block, WARP_SIZE x WARPS_IN_Y_DIR threads; the device code hard-codes these
// dimensions (dim_x, dim_y). Grid: 1 x (co-resident blocks / grid_divisor). A cooperative launch requires all
// blocks to be resident at once; the kernels grid-stride over their virtual blocks, so any
// grid.y >= 1 is correct and only the speed depends on it.
// fcn: a bfs_kernel instantiation from select_kernel(); grid_divisor: config_t::m_grid_divisor.
launch_geometry_t launch_geometry(void *fcn, int N, int R, int grid_divisor)
{
    // Smallest leaf: WARP_SIZE * 8 = 256 halves (a warp's row of half8 accesses). The device code
    // tiles every level in blocks of that width: with a smaller leaf tiles_count_x is 0 in the
    // deepest level and opt_mod divides by 0.
    CV_Assert((N >> R) >= WARP_SIZE * int(sizeof(half8) / sizeof(half)));

    // blockDim.x must be a multiple of warpSize: WARPS_IN_Y_DIR warps of one virtual block
    launch_geometry_t g;
    g.block.x = WARP_SIZE;
    g.block.y = BFSCaps::WARPS_IN_Y_DIR;

    if (REPORT_STATISTICS) {
        // diagnostic output only: cy / cx are not used for the launch
        // we need y=[N/2], x=[N/4] threads to run
        const auto M = N >> 1;
        const auto cy = N / 2 / g.block.y;
        const auto cx = N / 4 / g.block.x;
        std::cerr << "Using " << "half8 point arithmetic"
                  << std::endl;
        std::cerr << "Grid config: " << M << "x" << M << std::endl;
        std::cerr << "Algo config: N=" << N << ", R=" << R << std::endl;
        std::cerr << "Minimal mult block size: M=" << (N >> R) << std::endl;
        std::cerr << "Using block size: " << g.block.y << "x" << g.block.x << std::endl;
        std::cerr << "Prefered grid dim: " << cy << "x" << cx << std::endl;
    }
    dump_best_config(fcn); // prints occupancy hints only with REPORT_STATISTICS

    // blocks per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor) x SM count
    const auto totalActiveBlocks = cooperative_launch_config(fcn, g.block, true);
    g.grid.x = 1;
    g.grid.y = totalActiveBlocks / grid_divisor;
    CV_Assert(g.grid.y > 0);
    return g;
}

// First index and count of the nodes on layer R - 1 (the parents of the leaf multiplications).
// Breadth-first order: level l starts at 1 + 7 + ... + 7^(l - 1) and has 7^l nodes.
void leaf_parent_nodes(int R, unsigned int &m_nodes_start, unsigned int &m_nodes_count)
{
    m_nodes_start = 0;
    m_nodes_count = 1;
    for (auto l = 1; l < R; ++l) {
        m_nodes_start = m_nodes_start + m_nodes_count;
        m_nodes_count = m_nodes_count * MULTS_COUNT;
    }
}

// All loop counts must be powers of two that divide the leaf size: each loop handles N / loops
// rows of the top layer and (N >> l) / loops rows on layer l, down to the leaf blocks. For a leaf
// k * 256 with k not a power of two (N = k * 2^R), loops <= leaf is not enough (512 does not
// divide 768).
[[maybe_unused]] bool valid_loops(unsigned int loops, int N, int R)
{
    const auto leaf = static_cast<unsigned int>(N >> R);
    return loops >= 1 && 0 == (loops & (loops - 1)) && 0 == (leaf & (loops - 1));
}

[[noreturn]] void no_kernel(int operation, int N, int R)
{
    CV_Error(cv::Error::StsNotImplemented,
             "no bfs_kernel instantiated for operation=" + std::to_string(operation) + ", N="
                 + std::to_string(N) + ", R=" + std::to_string(R)
                 + " (add it to the select_kernel() table in bfs_caps.cu)");
}

} // namespace

/**
 * Launches bfs_kernel<operation> for (N, R) on `stream` (asynchronous). Called through
 * BFSCaps::caps<operation>() (forward_pass(), backward_pass(), ...), which skips R = 0. Requires the
 * address tables of initialize() (store_address_cache()). Throws cv::Exception if no kernel is
 * instantiated for (N, R) or, with run-time loop counts, a loop count in m_config is invalid.
 */
template<int operation>
void BFSCaps::process(int N, int R, cv::cuda::Stream &stream)
{
    // every flag combination fits into 5 bits (DECODE = 16 is the highest flag)
    CV_Assert(1 <= operation && 32 > operation);
    auto fcn = select_kernel<operation>(N, R);
    if (nullptr == fcn) {
        no_kernel(operation, N, R);
    }
    const auto geometry = launch_geometry(fcn, N, R, int(m_config.m_grid_divisor));

    // m node - multiplication nodes: the level R - 1 nodes, parents of the leaf products
    // (used by the reconstruction)
    unsigned int m_nodes_start, m_nodes_count;
    leaf_parent_nodes(R, m_nodes_start, m_nodes_count);

#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
    // kernel arguments in the order of the bfs_kernel parameters
    void *args[] = {&m_nodes_start, &m_nodes_count};
#else
    // loop counts come from m_config (heuristic from initialize(), or the result of tune())
    unsigned int encode_loops = m_config.m_transform_A_and_B_loops_count;
    unsigned int bilinear_loops = m_config.m_bilinear_loops_count;
    unsigned int reconstruct_loops = m_config.m_reconstruct_loops_count;
    unsigned int decode_loops = m_config.m_transform_C_loops_count;
    for (auto loops : {encode_loops, bilinear_loops, reconstruct_loops, decode_loops}) {
        if (!valid_loops(loops, N, R)) {
            CV_Error(cv::Error::StsOutOfRange,
                     "invalid loop count " + std::to_string(loops) + " for N=" + std::to_string(N)
                         + ", R=" + std::to_string(R));
        }
    }
    void *args[] = {&encode_loops,
                    &bilinear_loops,
                    &reconstruct_loops,
                    &decode_loops,
                    &m_nodes_start,
                    &m_nodes_count};
#endif
    cooperative_launch(fcn, args, geometry.grid, geometry.block, stream);
}

// ---------------------------------------------------------------------------------------------
// Tuning
//
// Loop counts are kernel arguments only when LOOPS_COUNT_AS_TEMPLATE_ARGS=OFF, so tuning needs
// such a build. Workflow:
//   1. build with -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF
//   2. ./bin/test_diagnostics tune            (writes this GPU's entries into gen/tuning_table.hpp)
//   3. rebuild with ON: select_kernel() compiles the table in (include/tuning.hpp)
// Tuning runs the kernels on the matrices passed to initialize() and overwrites them
// (encode/decode work in place): re-upload inputs before using the BFS object afterwards.
// ---------------------------------------------------------------------------------------------

// Tunes the row loop count of one phase. operation: ENCODE, BILINEAR, FULL_RECONSTRUCT or DECODE.
// Runs bfs_kernel<operation> for loops = 1, 2, 4, ..., N >> R (warm_ups untimed launches, then
// `iterations` timed ones, on a private stream) and returns a copy of m_config with that phase's
// loop count set to the fastest one and the derived rows-per-loop fields updated; m_config itself
// is not changed. The loop counts of the other phases are passed but unused, since the kernel runs
// only `operation`. Uses the grid of process() (m_config.m_grid_divisor).
BFSCaps::config_t BFSCaps::select_best_options(unsigned int operation, int warm_ups, int iterations)
{
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
    (void) operation;
    (void) warm_ups;
    (void) iterations;
    std::cerr << "BFS tuning needs a build with -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF "
                 "(loop counts are compiled into the kernel table)"
              << std::endl;
    return m_config;
#else
    cv::cuda::Stream stream;
    const auto N = get_dimension();
    const auto R = get_recursions();
    const auto leaf = static_cast<unsigned int>(N >> R);

    // the kernel to time and the config_t field it is tuned for
    void *fcn = nullptr;
    unsigned int config_t::*field = nullptr;
    const char *name = "";
    switch (operation) {
    case BFSCaps::ENCODE:
        fcn = select_kernel<BFSCaps::ENCODE>(N, R);
        field = &config_t::m_transform_A_and_B_loops_count;
        name = "encode";
        break;
    case BFSCaps::BILINEAR:
        fcn = select_kernel<BFSCaps::BILINEAR>(N, R);
        field = &config_t::m_bilinear_loops_count;
        name = "bilinear";
        break;
    case BFSCaps::FULL_RECONSTRUCT:
        fcn = select_kernel<BFSCaps::FULL_RECONSTRUCT>(N, R);
        field = &config_t::m_reconstruct_loops_count;
        name = "reconstruct";
        break;
    case BFSCaps::DECODE:
        fcn = select_kernel<BFSCaps::DECODE>(N, R);
        field = &config_t::m_transform_C_loops_count;
        name = "decode";
        break;
    default:
        CV_Error(cv::Error::StsBadArg,
                 "tuning supports ENCODE, BILINEAR, FULL_RECONSTRUCT and DECODE, got "
                     + std::to_string(operation));
    }
    if (nullptr == fcn) {
        no_kernel(int(operation), N, R);
    }
    const auto geometry = launch_geometry(fcn, N, R, int(m_config.m_grid_divisor));

    unsigned int m_nodes_start, m_nodes_count;
    leaf_parent_nodes(R, m_nodes_start, m_nodes_count);

    auto cfg = m_config;
    auto best_ms = std::numeric_limits<double>::max();
    auto best_loops = 0u;
    for (auto loops = 1u; loops <= leaf; loops *= 2) {
        // only the loop count of the tuned operation matters, the others are not used
        config_t trial = cfg;
        trial.*field = loops;
        void *args[] = {&trial.m_transform_A_and_B_loops_count,
                        &trial.m_bilinear_loops_count,
                        &trial.m_reconstruct_loops_count,
                        &trial.m_transform_C_loops_count,
                        &m_nodes_start,
                        &m_nodes_count};

        for (auto w = 0; w < warm_ups; ++w) {
            cooperative_launch(fcn, args, geometry.grid, geometry.block, stream);
        }
        stream.waitForCompletion();

        cuperf_t perf(std::string("tune_") + name);
        perf.start_timer(stream);
        for (auto i = 0; i < iterations; ++i) {
            cooperative_launch(fcn, args, geometry.grid, geometry.block, stream);
        }
        perf.end_timer(stream);
        perf.set_count(iterations);
        stream.waitForCompletion();

        const double ms = perf.get_avg();
        std::cerr << "  tune " << name << " N=" << N << " R=" << R << " loops=" << loops << ": "
                  << ms << " ms" << std::endl;
        if (ms < best_ms) {
            best_ms = ms;
            best_loops = loops;
        }
    }
    CV_Assert(best_loops > 0);
    cfg.*field = best_loops;
    cfg.m_bilinear_rows_per_loop = N / cfg.m_bilinear_loops_count;
    cfg.m_reconstruct_rows_per_loop = leaf / cfg.m_reconstruct_loops_count;
    std::cerr << "best " << name << " loops for N=" << N << ", R=" << R << ": " << best_loops
              << " (" << best_ms << " ms)" << std::endl;
    return cfg;
#endif
}

// Tunes the four phases one after another, each independently (select_best_options() keeps the
// other phases at m_config), prints the resulting GEMM_TUNED_BFS(...) line for
// gen/tuning_table.hpp to stdout and returns the result; apply = true also stores it in m_config.
// In a LOOPS_COUNT_AS_TEMPLATE_ARGS build it only prints why tuning is impossible.
BFSCaps::config_t BFSCaps::tune(int warm_ups, int iterations, bool apply)
{
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
    return select_best_options(BFSCaps::ENCODE, warm_ups, iterations); // prints the explanation
#else
    const auto N = get_dimension();
    const auto R = get_recursions();
    config_t best = m_config;
    const std::pair<unsigned int, unsigned int config_t::*> operations[] = {
        {BFSCaps::ENCODE, &config_t::m_transform_A_and_B_loops_count},
        {BFSCaps::BILINEAR, &config_t::m_bilinear_loops_count},
        {BFSCaps::FULL_RECONSTRUCT, &config_t::m_reconstruct_loops_count},
        {BFSCaps::DECODE, &config_t::m_transform_C_loops_count},
    };
    for (const auto &op : operations) {
        const auto cfg = select_best_options(op.first, warm_ups, iterations);
        best.*(op.second) = cfg.*(op.second);
    }
    best.m_bilinear_rows_per_loop = N / best.m_bilinear_loops_count;
    best.m_reconstruct_rows_per_loop = (N >> R) / best.m_reconstruct_loops_count;

    // the gen/tuning_table.hpp line (test_diagnostics tune writes it)
    std::cout << "GEMM_TUNED_BFS(\"" << tuning::device_name() << "\", " << N << ", " << R << ", "
              << best.m_transform_A_and_B_loops_count << ", " << best.m_bilinear_loops_count << ", "
              << best.m_reconstruct_loops_count << ", " << best.m_transform_C_loops_count << ")" << std::endl;
    if (apply) {
        m_config = best;
    }
    return best;
#endif
}

// explicit instantiations of BFSCaps::process<operation>; must follow the template definitions above
#include "bfs_caps.ini"
