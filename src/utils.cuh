/**
 * @file utils.cuh
 * @brief Device helpers shared by the BFS (CAPS) passes: half8 arithmetic, cached / streaming
 *        global loads and stores, and cheap power-of-two division / modulo.
 * @author GEMM Project
 * @date 2025
 * 
 * Included by alternative_basis.cuh (encoder / decoder), bilinear.cuh (forward sums S1 .. S6)
 * and reconstruct.cuh (backward pass), which are all compiled into bfs_kernel in src/bfs_caps.cu.
 *
 * Contents:
 * - operator+ / operator- for half8 (= float4 holding 8 halves, see core.hpp): lane-wise, as
 *   four half2 SIMD operations;
 * - load / store wrappers around the PTX cache-operator intrinsics. Every pass has a
 *   compile-time flag `cache` (bfs_kernel takes it from L2_CACHE_POLICY in core.hpp) that
 *   selects between "cache in L2, not L1" (__ldcg / __stcg) and "streaming, evict first"
 *   (__ldcs / __stcs); fetch_with_policy() / store_with_policy() add the LAST_USE_HINTS
 *   variant (__ldlu / __stcs) for the last access of an element in a pass;
 * - opt_div() / opt_mod(): division and modulo by a power of two (shift / mask), used for the
 *   block-id -> tile -> node index arithmetic of all passes.
 *
 * All templates take T = half2 or half8; pointers must be aligned to sizeof(T).
 */

#pragma once

#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include <cooperative_groups.h>
#include <core.hpp>

// Only for IDEs / host-side parsers that read this header without nvcc: provide dummy
// built-in variables and __threadfence() so that the device code parses. nvcc defines
// __CUDACC__, so none of this is compiled in a real build (the .cuh files that include this one
// also define __CUDACC__ themselves before the include).
#ifndef __cplusplus
#define __cplusplus
#endif

#ifndef __CUDACC__
/** @brief Block index for host compilation */
dim3 blockIdx;
/** @brief Block dimensions for host compilation */
dim3 blockDim;
/** @brief Thread index for host compilation */
dim3 threadIdx;
/** @brief Grid dimensions for host compilation */
dim3 gridDim;

/** @brief Thread fence for host compilation */
inline void __threadfence() {}

#define __CUDACC__
#endif

/**
 * @brief Calculate ceiling division
 * @param a Dividend
 * @param b Divisor
 * @return Ceiling of a/b
 * 
 * Computes the ceiling division of two integers, useful for calculating
 * grid and block dimensions in CUDA kernels.
 */
__host__ __device__
inline auto div_ceil(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/**
 * @brief Lane-wise sum of two half8 values (8 halves each).
 * @details half8 is a float4; each of its 32-bit components x, y, z, w is reinterpreted as a
 * half2, so the sum is four __hadd2 instructions. Result in half precision (no float accumulation).
 */
__device__ __forceinline__ auto operator+(const half8 &a, const half8 &b)
{
    half8 r;
    auto result = reinterpret_cast<half2 *>(&r);
    {
        result[0] = *reinterpret_cast<const half2 *>(&a.x) + *reinterpret_cast<const half2 *>(&b.x);
        result[1] = *reinterpret_cast<const half2 *>(&a.y) + *reinterpret_cast<const half2 *>(&b.y);
        result[2] = *reinterpret_cast<const half2 *>(&a.z) + *reinterpret_cast<const half2 *>(&b.z);
        result[3] = *reinterpret_cast<const half2 *>(&a.w) + *reinterpret_cast<const half2 *>(&b.w);
    }
    return r;
}

/**
 * @brief Lane-wise difference a - b of two half8 values (four half2 subtractions, see operator+).
 */
__device__ __forceinline__ auto operator-(const half8 &a, const half8 &b)
{
    half8 r;
    auto result = reinterpret_cast<half2 *>(&r);
    {
        result[0] = *reinterpret_cast<const half2 *>(&a.x) - *reinterpret_cast<const half2 *>(&b.x);
        result[1] = *reinterpret_cast<const half2 *>(&a.y) - *reinterpret_cast<const half2 *>(&b.y);
        result[2] = *reinterpret_cast<const half2 *>(&a.z) - *reinterpret_cast<const half2 *>(&b.z);
        result[3] = *reinterpret_cast<const half2 *>(&a.w) - *reinterpret_cast<const half2 *>(&b.w);
    }
    return r;
}

namespace cg = cooperative_groups;

/**
 * @brief Load through the read-only (non-coherent, texture) data path: __ldg = ld.global.nc.
 * @details Despite the name, this does not bypass L2 (every global access goes through L2);
 * what it changes is that the line may be cached in L1 / the read-only cache. Only valid for
 * data that is not written during the kernel. Used by reconstruct.cuh when `cache` is false.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ T fetch_bypass_L2(const T *__restrict__ A)
{
    // read-only path (ld.global.nc); the data must not be modified while the kernel runs
    return __ldg(A);
}

/**
 * @brief Streaming load: __ldcs = ld.global.cs ("cache streaming", likely accessed once).
 * @details The line is allocated with evict-first priority in L1 and L2, so it does not push
 * out data that is still going to be reused. Used when the pass's `cache` flag is false.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ T fetch_no_cache(const T *__restrict__ A)
{
    // not expected to be reused soon (one-time use): streaming hint, evict first
    return __ldcs(A);
}

/**
 * @brief L2-cached load: __ldcg = ld.global.cg (cache in L2 and below, not in L1).
 * @details Used when the pass's `cache` flag is true: the encoded matrices, sums and products are
 * re-read by the next level or pass, usually by another block / SM, so they are kept in L2
 * without allocating L1 lines that the same thread would never hit again.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ T fetch_cache_L2(const T *__restrict__ A)
{
    // Cache at global level (cache in L2 and below, not L1).
    return __ldcg(A);
}

/**
 * @brief Default store: __stwb = st.global.wb (write-back through all coherent cache levels).
 * @details Not used by the BFS passes at the moment (they use store_cache_L2() /
 * store_no_cache() / store_with_policy()); kept as the plain alternative.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ void store(T *__restrict__ A, const T& value)
{
    // default store
    //__stcs(A, value);
    // write-back policy: cached on the way and eventually written to global memory
    __stwb(A, value);
}

/**
 * @brief Streaming store: __stcs = st.global.cs (evict-first, likely not read again soon).
 * @details The `cache == false` counterpart of store_cache_L2(); also the LAST_USE_HINTS store.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ void store_no_cache(T *__restrict__ A, const T& value)
{
    // streaming data: allocate with evict-first priority so it does not displace reused lines
    __stcs(A, value);
    //__stwb(A, value);
}

/**
 * @brief L2-cached store: __stcg = st.global.cg (cache in L2, not in L1).
 * @details Used when the pass's `cache` flag is true: the value is read again by the next
 * level or pass (possibly by another SM), so it should stay in L2.
 */
template<class T> // half2 or half8
static __device__ __forceinline__ void store_cache_L2(T *__restrict__ A, const T& value)
{
    // Cache at global level (cache in L2 and below, not L1)
    __stcg(A, value);
}

/**
 * @brief Load with the pass's cache policy, or as a last-use load when @p last is set.
 * @tparam cache true: __ldcg (fetch_cache_L2), false: __ldcs (fetch_no_cache)
 * @param last true when this is the last access of the element in the current pass (its rows
 *             are not touched again), computed by the caller only when LAST_USE_HINTS is on
 *             (otherwise a compile-time false and the branch disappears)
 * @details LAST_USE_HINTS: the last access is a last-use load (__ldlu = ld.global.lu), so the
 * line can leave the cache before data that is still needed. Callers use it only for operands
 * that are read and not written by the same access; the written ones use store_with_policy().
 */
template<class T, bool cache>
static __device__ __forceinline__ T fetch_with_policy(const T *__restrict__ A, bool last)
{
    if (last) {
        return __ldlu(A);
    }
    return cache ? fetch_cache_L2(A) : fetch_no_cache(A);
}

/**
 * @brief Store with the pass's cache policy, or as an evict-first store when @p last is set.
 * @tparam cache true: __stcg (store_cache_L2), false: __stcs (store_no_cache)
 * @param last see fetch_with_policy(); when set the store is __stcs (st.global.cs). With
 *             cache == false this is the same instruction as the normal store.
 */
template<class T, bool cache>
static __device__ __forceinline__ void store_with_policy(T *__restrict__ A, const T &value, bool last)
{
    if (last) {
        __stcs(A, value);
    } else if (cache) {
        store_cache_L2(A, value);
    } else {
        store_no_cache(A, value);
    }
}

/** @brief Load with the pass's cache policy (no last-use variant): __ldcg if cache, else __ldcs. */
template<class T, bool cache>
static __device__ __forceinline__ auto read(const T *__restrict__ A)
{
    return cache ? fetch_cache_L2(A) : fetch_no_cache(A);
}

/**
 * @brief a / b for a power of two b > 0, as a right shift (two instructions: bfind + shr).
 * @details POW2 (the default): b is a power of two; for other b the result is a >> floor(log2(b)),
 * and for b == 0 bfind returns 0xFFFFFFFF. The divisors of the BFS passes (tile counts, block
 * sizes N >> l, rows per node) are powers of two when N is (pow2_v<N>); otherwise POW2 = false
 * divides.
 * (The "complexity = 5" notes in the passes are rough op-count estimates for this call.)
 */
template<bool POW2 = true>
__device__ __forceinline__ unsigned int opt_div(unsigned int a, unsigned int b)
{
    if constexpr (POW2) {
        // b must be a power of 2: a / b == a >> log2(b). bfind gives log2(b) in one instruction
        // (__ffs(b) - 1 and 31 - __clz(b) are equivalent but cost an extra op).
        unsigned int shift;
        asm("bfind.u32 %0, %1;" : "=r"(shift) : "r"(b));
        return a >> shift;
    } else {
        return a / b; // N = k * 2^R with k not a power of two (b is a constant of the instantiation where it can be)
    }
}

/**
 * @brief a % b for a power of two b > 0, as a mask: a & (b - 1).
 * @details Precondition: b is a power of two (wrong results otherwise). "complexity = 2": one
 * subtraction and one AND.
 */
template<bool POW2 = true>
__device__ __forceinline__ unsigned int opt_mod(unsigned int a, unsigned int b)
{
    if constexpr (POW2) {
        return a & (b - 1); // b must be a power of 2
    } else {
        return a % b;
    }
}

/// Whether the BFS divisors of a size-N instantiation (tile counts, block sizes N >> l, rows per
/// node) are powers of two: exactly when N is (N = k * 2^R, the leaf size k a multiple of 256)
template<unsigned int N>
constexpr bool pow2_v = (N & (N - 1)) == 0;
