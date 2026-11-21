/**
 * @file alternative_basis.cuh
 * @brief Encoder / decoder passes of CAPS: change of basis of A, B (before) and C (after).
 * @author GEMM Project
 * @date 2025
 * 
 * CAPS multiplies in an alternative basis (Karstadt-Schwartz): A and B are first transformed in
 * place, then the forward sums (bilinear.cuh), the leaf GEMMs and the backward combination
 * (reconstruct.cuh) need fewer additions, and at the end the result is transformed back.
 * bfs_kernel (src/bfs_caps.cu) runs these passes:
 *   BFSCaps::ENCODE: alternative_basis_transform_kernel<encoder_functor_t> on A and B (in place);
 *   BFSCaps::DECODE: alternative_basis_transform_kernel<decoder_functor_t> on the product, which the
 *                backward pass left in the top N x N rows of the root's S buffer (in place).
 *
 * The transform is recursive: for every level l = 1 .. R the matrix is cut into blocks of
 * 2M x 2M halves (M = N >> l), each seen as
 *     a b
 *     c d          (quadrants of M x M halves)
 * and one elementwise operation is applied per block:
 *     encoder (A and B):  d <- b + d - c   (A22 = A12 - A21 + A22)
 *     encoder with transform == true:                d <- c + d - b
 *     decoder:                                       b <- b - d,  c <- d - c
 * Quadrant a is never touched. The per-level operations act on different bits of the element
 * index, so the order of the levels does not change the result, but each level must finish
 * (grid barrier) before the next one reads the same elements.
 *
 * Matrices: N x N halves, row-major, continuous, leading dimension N (ldab). Each thread
 * handles one T (half8, i.e. 8 consecutive halves of a row) of a d quadrant and
 * reads / writes the matching elements of b (M rows up) and c (M columns left).
 */

#pragma once

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.cuh"
#include "bfs_caps.hpp"

/**
 * @brief Encoder functor: d <- b + d - c on one half8 of a 2M x 2M block a b / c d (see the file
 *        comment; A22 = A12 - A21 + A22). B is encoded like A.
 * @tparam cache true: loads / stores cached in L2 (__ldcg / __stcg), false: streaming
 *               (__ldcs / __stcs); bfs_kernel passes L2_CACHE_POLICY, DFSCaps true
 *
 * Interface used by alternative_basis_transform_kernel (and DFSCaps' transform_level_kernel):
 * typedef T, and runLR(A, B, ldab, N, last), which encodes the element of A and of B at the same
 * offset. A points at the element of the d quadrant; ldab in halves; N = M, the quadrant size;
 * last: last access of these rows in the pass (LAST_USE_HINTS). 16-byte aligned addresses.
 */
// complexity(op) = 2 * 8 half2 operations, complexity(communication) = 2 * (3 + 1) * sizeof(half8)
template<bool cache>
struct encoder_functor_t
{
    typedef half8 T;
    /** @brief Encode one half8 of a d quadrant in place: b is M rows up, c M columns left */
    static __device__ __forceinline__ void run(half *__restrict__ A, unsigned int ldab, unsigned int N, bool last)
    {
        const auto *b_ptr = reinterpret_cast<const T *>(A - size_t(N) * ldab);
        const auto *c_ptr = reinterpret_cast<const T *>(A - N);
        const auto b = fetch_with_policy<T, cache>(b_ptr, last);
        const auto c = fetch_with_policy<T, cache>(c_ptr, last);
        const auto d = read<T, cache>(reinterpret_cast<const T *>(A));
        store_with_policy<T, cache>(reinterpret_cast<T *>(A), b + d - c, last);
    }

    static __device__ __forceinline__ void runLR(half *__restrict__ A, half *__restrict__ B, unsigned int ldab,
                                                 unsigned int N, bool last)
    {
        run(A, ldab, N, last);
        run(B, ldab, N, last);
    }
};

/**
 * @brief Decoder functor: transforms the product C back from the alternative basis.
 * @tparam cache as for encoder_functor_t
 *
 * Per 2M x 2M block (a b / c d), one half8 per call:
 *     A12 -= A22       (b <- b - d)
 *     A21 = A22 - A21  (c <- d - c)
 * d and a are unchanged. runLR(A, B, ...) has the encoder's signature so that
 * alternative_basis_transform_kernel can run both; B is ignored (bfs_kernel passes the same
 * pointer, the root's S buffer, as A and B).
 */
// complexity(op) = 8 half2 operations, complexity(communication) = (3 + 2) * sizeof(half8)
template<bool cache>
struct decoder_functor_t
{
    typedef half8 T;
    static __device__ __forceinline__ void runLR(half *__restrict__ A, half *__restrict__, unsigned int ldab,
                                                 unsigned int N, bool last)
    {
        auto *b_ptr = reinterpret_cast<T *>(A - size_t(N) * ldab);
        auto *c_ptr = reinterpret_cast<T *>(A - N);
        const auto b = read<T, cache>(b_ptr);
        const auto c = read<T, cache>(c_ptr);
        const auto d = fetch_with_policy<T, cache>(reinterpret_cast<const T *>(A), last);
        store_with_policy<T, cache>(b_ptr, b - d, last);
        store_with_policy<T, cache>(c_ptr, d - c, last);
    }
};

/**
 * @brief Map a (virtual) thread to one T of a d quadrant at the current level.
 * @tparam dim_x tile width in halves (WARP_SIZE * DIM_SCALE: one warp row of T accesses)
 * @tparam dim_y tile height in rows (BFSCaps::WARPS_IN_Y_DIR)
 * @param id         virtual block id (blockIdx.y + k * gridDim.y, grid-stride loop)
 * @param alternative_basis_loops number of row loops of the pass (rows per thread step)
 * @param x          column of the thread inside the tile, in halves (threadIdx.x * DIM_SCALE)
 * @param y          row of the thread inside the tile (threadIdx.y)
 * @param max_y      N / 2: height (and width) of the virtual space
 * @param tiles_count_x tiles per virtual row: (N / 2) / dim_x (power of two)
 * @param block_size quadrant size M = N >> l in halves (power of two)
 * @return x, y: column and row (in halves) of the element in the full N x N matrix, without
 *         the row-loop offset (the caller adds `row`); z: != 0 when the virtual position is
 *         outside the (N/2) x (N/2) space and the thread has nothing to do
 *
 * Every level touches exactly (N/2)^2 elements: the d quadrants of all (N / 2M)^2 blocks of
 * 2M x 2M. They are enumerated as a virtual (N/2) x (N/2) matrix, cut into M x M tiles; tile
 * (block_idx, block_idy) is the d quadrant of block (block_idx, block_idy), which starts at
 * (2 * block_id + 1) * M in the real matrix. Virtual rows are multiples of
 * alternative_basis_loops: loop iteration `row` processes the rows congruent to `row` modulo
 * the loop count, so one launch step covers (N/2) x (N/2 / loops) elements.
 */
// process each R row
template<unsigned int dim_x, unsigned int dim_y, bool POW2>
__device__ __forceinline__ auto calculate_alternative_basis_virtual_address(
    unsigned int id,
    unsigned int alternative_basis_loops,
    unsigned int x,
    unsigned int y,
    unsigned int max_y,
    unsigned int tiles_count_x,
    unsigned int block_size)
{
    /**
     * calc virtual pos - 40
     * memory ops = 3 + 1 or 3 + 2
     * arithmetic ops = 2 * 11
     */

    // tile id in x dir, complexity = 2
    const auto tile_id_x = opt_mod<POW2>(id, tiles_count_x);

    // tile id in y dir, complexity = 5
    const auto tile_id_y = opt_div<POW2>(id, tiles_count_x);

    // generate synthetic x and y, complexity = 2 * 2
    // xx in [0, N/2) halves; yy a multiple of alternative_basis_loops
    const auto xx = tile_id_x * dim_x + x;
    const auto yy = (tile_id_y * dim_y + y) * alternative_basis_loops;

    // calculate node idx, complexity = 5: 0 inside the virtual space, > 0 past its last row
    const auto node_idx = opt_div<POW2>(yy, max_y);

    const auto xxx = xx;
    const auto yyy = yy;

    // index of the 2M x 2M block (= of its d quadrant) in x, complexity = 5
    const auto block_idx = opt_div<POW2>(xxx, block_size);
    // index of the 2M x 2M block in y, complexity = 5
    const auto block_idy = opt_div<POW2>(yyy, block_size);

    // position inside the d quadrant, complexity = 2 * 2
    const auto tx = opt_mod<POW2>(xxx, block_size);
    const auto ty = opt_mod<POW2>(yyy, block_size);

    // start of the d quadrant of the block: block * 2M + M, complexity = 2 * 3
    const auto block_start_x = (block_idx << 1) * block_size + block_size;
    const auto block_start_y = (block_idy << 1) * block_size + block_size;

    // element position in the full matrix, complexity = 2 * 1
    uint3 s;
    s.x = block_start_x + tx;
    s.y = block_start_y + ty;
    s.z = node_idx;
    return s;
}

/**
 * @brief Encoder / decoder pass: applies @p transformer to all levels 1 .. R, in place.
 * @tparam N           matrix size in halves (N x N, leading dimension N)
 * @tparam R           number of recursion levels
 * @tparam transformer encoder_functor_t<cache> or decoder_functor_t<cache>
 * @param barrier      grid group of the cooperative launch of bfs_kernel
 * @param A, B         matrices to transform (A = left, B = right operand; the decoder gets the
 *                     product as A and ignores B)
 * @param alternative_basis_loops row loops (template argument with LOOPS_COUNT_AS_TEMPLATE_ARGS)
 *
 * Loop structure: for each row class `row` (rows congruent to `row` mod loops) and each level
 * l = 1 .. R, the grid strides over the virtual (N/2) x (N/2 / loops) space (see
 * calculate_alternative_basis_virtual_address()), each thread transforms one T of a d quadrant,
 * then a grid-wide barrier separates the levels. Row classes are independent (b and c are M
 * rows / columns away and M is a multiple of the loop count), so splitting the pass into loops
 * only shrinks the working set of each step. Assumes power-of-two sizes and loops <= N >> R
 * (otherwise va.y + row would leave the quadrant). Block shape: WARP_SIZE x WARPS_IN_Y_DIR
 * threads used per tile (threadIdx.x, threadIdx.y); only blockIdx.y and gridDim.y are used.
 */
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
template<unsigned int N, unsigned int R, class transformer, unsigned int alternative_basis_loops>
__device__ __forceinline__ void alternative_basis_transform_kernel(cg::grid_group &barrier,
                                                                   half *__restrict__ A,
                                                                   half *__restrict__ B)
#else
template<unsigned int N, unsigned int R, class transformer>
__device__ __forceinline__ void alternative_basis_transform_kernel(
    cg::grid_group &barrier,
    half *__restrict__ A,
    half *__restrict__ B,
    unsigned int alternative_basis_loops)
#endif
{
    typedef typename transformer::T T;
    constexpr auto DIM_SCALE = sizeof(T) / sizeof(half);

    // column inside the tile in halves: each thread covers DIM_SCALE halves,
    // so x in [0, WARP_SIZE * DIM_SCALE)
    const unsigned int x = threadIdx.x * DIM_SCALE;
    const unsigned int y = threadIdx.y;
    const unsigned int total_blocks_count = gridDim.y;

    // A and B are continuous N x N matrices, so the leading dimension is N (in halves)
    constexpr auto ldab = N;

    constexpr auto dim_x = WARP_SIZE * DIM_SCALE;
    constexpr auto dim_y = BFSCaps::WARPS_IN_Y_DIR;

    // every level updates (N/2) x (N/2) elements (all d quadrants), the virtual space size
    constexpr auto max_x = N >> 1;
    constexpr auto max_y = max_x;

    // tiles count in x dir
    constexpr auto tiles_count_x = max_x / dim_x;

GEMM_ROW_LOOP_UNROLL
    for (auto row = 0; row < alternative_basis_loops; ++row) {
#pragma unroll
        for (auto l = 1; l <= R; ++l) {
            // quadrant size at level l: blocks of 2M x 2M, quadrants of M x M
            const auto M = N >> l;
            const auto block_size = M;

            // grid-stride loop over the virtual tiles; exits through the break below
            unsigned int block_start_id = blockIdx.y;
            do {
                /**
                 * calc virtual pos - 40
                 * memory ops = 3 + 1 or 3 + 2
                 * arithmetic ops = 2 * 11
                 */

                const auto id = block_start_id;
                const auto va = calculate_alternative_basis_virtual_address<
                    dim_x,
                    dim_y,
                    pow2_v<N>>(id, alternative_basis_loops, x, y, max_y, tiles_count_x, block_size);

                // past the end of the virtual space: this thread is done with level l, complexity = 1
                if (0 < va.z) {
                    break;
                }

                // offset in global memory space, complexity = 2
                {
                    // real row: virtual rows are multiples of the loop count, `row` selects the class
                    const auto sy = va.y + row;
                    // size_t: sy * N reaches 2^32 - 1 at N = 65536 and wraps above (N = 131072)
                    const auto offset = size_t(sy) * ldab + va.x;
                    auto A_ptr = A + offset;
                    auto B_ptr = B + offset;

                    // -- 40 + 2 ops + 22
                    // so in total we have 64 ops
                    // last: no later level of this pass touches the element. Levels l' > l only
                    // modify quadrants b, c, d of sub-blocks, so the elements whose offset inside
                    // the M x M quadrant is in its top-left (N >> R) x (N >> R) corner (quadrant a
                    // at every deeper level) are final now, and at l == R all are. b and c share
                    // d's offset modulo M, so one flag serves all three. va.y is a multiple of the
                    // loop count and so is N >> R, hence testing va.y instead of sy is equivalent.
#if LAST_USE_HINTS
                    const auto last = l == R || (opt_mod<pow2_v<N>>(va.x, block_size) < (N >> R)
                                                 && opt_mod<pow2_v<N>>(va.y, block_size) < (N >> R));
#else
                    constexpr auto last = false;
#endif
                    transformer::runLR(A_ptr, B_ptr, ldab, block_size, last);
                }
                block_start_id += total_blocks_count;
            } while (true);
            // level l must be complete everywhere before level l + 1 reads the same elements
            barrier.sync();
        }
    }
}
