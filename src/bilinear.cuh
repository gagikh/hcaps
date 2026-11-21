/**
 * @file bilinear.cuh
 * @brief Forward pass of CAPS: the six sums S1 .. S6 of every internal node.
 * @author GEMM Project
 * @date 2025
 * 
 * "Bilinear" refers to the bilinear (Strassen-Winograd type) algorithm, not to interpolation.
 * For a node with operands A, B of 2M x 2M halves (already in the alternative basis, see
 * alternative_basis.cuh) the 7 products are
 *     m1 = A11 * B11   m2 = A12 * B21   m3 = A21 * S4   m4 = A22 * B22
 *     m5 = S1 * S5     m6 = S2 * S6     m7 = S3 * B12
 * with the sums computed here:
 *     S1 = A21 + A22   S2 = A22 - A12   S3 = A22 - A11
 *     S4 = B22 - B11   S5 = B21 + B22   S6 = B22 - B12
 * The sums of a node of level l - 1 are the operands of its children at level l (see
 * BFSCaps::initialize() in src/bfs_caps.cpp), so bilinear_kernel (src/bfs_caps.cu) runs the levels top-down
 * with a grid barrier in between and calls bilinear<T, ...>::process() once per thread and T.
 *
 * Memory layout (set up by BFSCaps::initialize()): node operands are views into larger matrices with
 * the global leading dimension N halves; the S area of a node has 3M rows x 2M halves:
 *     S1 S2
 *     S3 S4
 *     S5 S6       (each M x M, same leading dimension N)
 * The node addresses come from the constant-memory table __intermediate_address_cache__
 * (BFSCaps::layer_info_t: m_L = A, m_R = B, m_S = S), indexed by node in BFS order.
 *
 * add_and_store / sub_and_store(_with_policy) are half8 helpers also used by reconstruct.cuh.
 */

#pragma once

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.cuh"
#include "bfs_caps.cuh"

/**
 * @brief *dst = a + b (half8, lane-wise), stored with the pass's cache policy.
 * @tparam cache true: __stcg (keep in L2), false: __stcs (streaming)
 */
// complexity(op) = 4
// complexity(communication) = 4 * sizeof(half8)
template<bool cache>
__device__ __forceinline__ void add_and_store(const half8 &a, const half8 &b, half8 *dst)
{
    auto src = a + b;
    cache ? store_cache_L2(dst, src) : store_no_cache(dst, src);
}

// add_and_store() with the LAST_USE_HINTS store: __stcs when last, else as add_and_store()
template<bool cache>
__device__ __forceinline__ void add_and_store_with_policy(const half8 &a, const half8 &b, half8 *dst, bool last)
{
    store_with_policy<half8, cache>(dst, a + b, last);
}

/**
 * @brief *dst = a - b (half8, lane-wise), stored with the pass's cache policy (see add_and_store).
 */
// complexity(op) = 4
// complexity(communication) = 4 * sizeof(half8)
template<bool cache>
__device__ __forceinline__ void sub_and_store(const half8 &a, const half8 &b, half8 *dst)
{
    auto src = a - b;
    cache ? store_cache_L2(dst, src) : store_no_cache(dst, src);
}

// sub_and_store() with the LAST_USE_HINTS store: __stcs when last, else as sub_and_store()
template<bool cache>
__device__ __forceinline__ void sub_and_store_with_policy(const half8 &a, const half8 &b, half8 *dst, bool last)
{
    store_with_policy<half8, cache>(dst, a - b, last);
}

/**
 * @brief Forward sums of one node, per thread one half8 of each of S1 .. S6.
 * @tparam cache true: L2-cached loads / stores (__ldcg / __stcg), false: streaming
 *               (bfs_kernel passes L2_CACHE_POLICY && R > 1: with R > 1 the sums of one level
 *               are read by the next level in the same kernel, with R == 1 only later by the
 *               leaf GEMMs)
 * @tparam calculate_A compute S1, S2, S3 (from A)
 * @tparam calculate_B compute S4, S5, S6 (from B); bfs_kernel sets both to true
 */
template<bool cache, bool calculate_A, bool calculate_B>
struct bilinear
{
    typedef half8 T;

    /**
     * @brief The six sums of one half8 position of a node: S1 = A21 + A22, S2 = A22 - A12,
     *        S3 = A22 - A11 (A = L), S4 = B22 - B11, S5 = B21 + B22, S6 = B22 - B12 (B = R), into
     *        the node's S area laid out S1 S2 / S3 S4 / S5 S6 (3M rows x 2M halves)
     * @param L, R, S the node's left / right operand and S area (all with leading dimension global_lds)
     * @param global_lds leading dimension in half8 (N / 8 for BFSCaps)
     * @param M quadrant size in halves
     * @param xx column inside the M x M quadrant in half8, 0 .. M / 8 - 1; yy: row, 0 .. M - 1
     * @param last last access in the pass (LAST_USE_HINTS)
     * Also used by DFSCaps' sums_kernel (src/dfs_caps.cu), whose nodes are not in the address cache.
     */
    static __device__ __forceinline__ void sums(const T *__restrict__ L,
                                                const T *__restrict__ R,
                                                T *__restrict__ S,
                                                unsigned int global_lds,
                                                unsigned int M,
                                                unsigned int xx,
                                                unsigned int yy,
                                                bool last)
    {
        constexpr auto DIM_SCALE = sizeof(T) / sizeof(half);
        const auto offset_x = M / DIM_SCALE;   // next quadrant to the right, in T
        const auto offset_abs = M * global_lds; // next quadrant down
        const auto shift = yy * global_lds + xx;

        auto S1 = S + shift;
        auto S2 = S1 + offset_x;
        auto S3 = S1 + offset_abs;
        auto S4 = S3 + offset_x;
        auto S5 = S3 + offset_abs;
        auto S6 = S5 + offset_x;

        if (calculate_A) {
            const auto A11 = L + shift;
            const auto a11 = fetch_with_policy<T, cache>(A11, last);
            const auto a12 = fetch_with_policy<T, cache>(A11 + offset_x, last);
            const auto a21 = fetch_with_policy<T, cache>(A11 + offset_abs, last);
            const auto a22 = fetch_with_policy<T, cache>(A11 + offset_abs + offset_x, last);
            add_and_store_with_policy<cache>(a21, a22, S1, last);
            sub_and_store_with_policy<cache>(a22, a12, S2, last);
            sub_and_store_with_policy<cache>(a22, a11, S3, last);
        }
        if (calculate_B) {
            const auto B11 = R + shift;
            const auto b11 = fetch_with_policy<T, cache>(B11, last);
            const auto b12 = fetch_with_policy<T, cache>(B11 + offset_x, last);
            const auto b21 = fetch_with_policy<T, cache>(B11 + offset_abs, last);
            const auto b22 = fetch_with_policy<T, cache>(B11 + offset_abs + offset_x, last);
            sub_and_store_with_policy<cache>(b22, b11, S4, last);
            add_and_store_with_policy<cache>(b21, b22, S5, last);
            sub_and_store_with_policy<cache>(b22, b12, S6, last);
        }
    }

    /// sums() of node idx of the BFS address cache (constant memory, broadcast: idx is uniform
    /// within a warp)
    static __device__ __forceinline__ void process(unsigned int idx,
                                                   unsigned int global_lds,
                                                   unsigned int M,
                                                   unsigned int xx,
                                                   unsigned int yy,
                                                   bool last)
    {
        const auto s = __intermediate_address_cache__[idx];
        sums(reinterpret_cast<const T *>(s.m_L), reinterpret_cast<const T *>(s.m_R), reinterpret_cast<T *>(s.m_S),
             global_lds, M, xx, yy, last);
    }
};
