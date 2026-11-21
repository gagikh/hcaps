/**
 * @file reconstruct.cuh
 * @brief Backward pass of CAPS: combining the 7 products of every node into its product.
 * @author GEMM Project
 * @date 2025
 * 
 * After the leaf GEMMs, each internal node's product C (2M x 2M, still in the alternative
 * basis) is built from its 7 products m1 .. m7 (M x M each, see bilinear.cuh for their
 * definitions):
 *     C11 = m1 + m2            C12 = m5 - m7
 *     C21 = m3 + m6            C22 = m5 + m6 - m2 - m4
 * bfs_kernel (src/bfs_caps.cu) runs it bottom-up:
 *   reconstruct_leaf_mults_kernel         -> block_reconstruct<T>::leaf_block_reconstruct() for
 *       the nodes of level R - 1, whose 7 products are the leaf GEMM outputs
 *       (__mult_address_cache__);
 *   reconstruct_intermediate_mults_kernel -> intermediate_block_reconstruct() for levels
 *       R - 2 .. 0, whose products are the reconstructed products of their children.
 * Each node's product is stored in the top 2M rows of its own S area (the sums S1 .. S4 are
 * dead by then), so after level 0 the full N x N result sits at the start of the root's S buffer,
 * where the decoder (alternative_basis.cuh) transforms it back to the standard basis.
 *
 * One call per thread handles one T (half8) of each of the four output quadrants.
 * xx is in T units, yy in rows, both inside an M x M quadrant.
 * The half8 code uses add_and_store / sub_and_store from bilinear.cuh, which is not included
 * here: src/bfs_caps.cu includes bilinear.cuh before this file.
 */

#pragma once

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.cuh"

/**
 * @brief Backward-pass combination of the 7 products of a node (see the file comment).
 * @tparam T Vector type of one access: half8 (the only specialization)
 *           (the primary template only declares the interface)
 * @tparam cache true: products stored with __stcg (kept in L2 for the next level / the decoder),
 *               false: __stcs; the loads differ per function, see the specializations
 */
template<class T, bool cache>
struct block_reconstruct
{
    /**
     * @brief Reconstruct the product of a node of level R - 1 from its 7 leaf products
     * @param s_node_idx Node index in __intermediate_address_cache__ (destination: its S area)
     * @param m_node_idx Index of its leaf-product buffer in __mult_address_cache__
     * @param global_lds Leading dimension of S in T units
     * @param M Leaf size in halves (N >> R)
     * @param xx Column inside the leaf product, in T units
     * @param yy Row inside the leaf product
     */
    static __device__ __forceinline__ void leaf_block_reconstruct(unsigned int s_node_idx,
                                                                  unsigned int m_node_idx,
                                                                  unsigned int global_lds,
                                                                  unsigned int M,
                                                                  unsigned int xx,
                                                                  unsigned int yy);

    /**
     * @brief Reconstruct the product of an internal node from its 7 children's products
     * @param s_node_idx Destination node index in __intermediate_address_cache__
     * @param m_node_idx Index of its first child (the 7 children are consecutive)
     * @param global_lds Leading dimension of the S areas in T units
     * @param M Child size in halves
     * @param xx Column inside the child product, in T units
     * @param yy Row inside the child product
     */
    static __device__ __forceinline__ void intermediate_block_reconstruct(unsigned int s_node_idx,
                                                                          unsigned int m_node_idx,
                                                                          unsigned int global_lds,
                                                                          unsigned int M,
                                                                          unsigned int xx,
                                                                          unsigned int yy);
};

/**
 * @brief Backward pass for half8 (8 halves, 16 bytes per access); the default configuration.
 * @details C11, C12, C21 use add/sub_and_store
 * (bilinear.cuh); C22 needs four terms and is computed as four half2 lanes.
 */
// 13 = 7 reads, 4 stores, 2 cache
// 41 ops
template<bool cache>
struct block_reconstruct<half8, cache>
{
    /**
     * @brief Combine the 7 leaf products of a node of level R - 1 into its product.
     * @param s_node_idx node (level R - 1) in __intermediate_address_cache__; its product
     *                   (2M x 2M) is written into the top 2M rows of its S area, i.e. over the
     *                   sums S1 .. S4, which the leaf GEMMs have consumed
     * @param m_node_idx the same node's entry in __mult_address_cache__: a continuous buffer of
     *                   the 7 leaf products m1 .. m7, each M x M halves with leading dimension M,
     *                   stacked vertically (m_i at rows (i - 1) * M)
     * @param global_lds leading dimension of S in T units (N / DIM_SCALE)
     * @param M          leaf size in halves (N >> R)
     * @param xx         column inside the leaf product in T units, 0 .. M / DIM_SCALE - 1
     * @param yy         row inside the leaf product, 0 .. M - 1
     * @details C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6, C22 = m5 + m6 - m2 - m4 (the
     * product is still in the alternative basis; the decoder converts it at the end).
     * The leaf products are read exactly once: __ldcs if cache, else __ldg (fetch_bypass_L2).
     * Both indices are uniform within a warp, so the constant-memory reads are broadcasts.
     */
    static __device__ __forceinline__ void leaf_block_reconstruct(unsigned int s_node_idx,
                                                                  unsigned int m_node_idx,
                                                                  unsigned int global_lds,
                                                                  unsigned int M,
                                                                  unsigned int xx,
                                                                  unsigned int yy)
    {
        // xx = 0, 1, 2, ... in half8 units
        // read from address cache
        // this is potential stall for the warp
        // TODO: use shared memory instead and only first warp will read this
        auto m = reinterpret_cast<const half8 *>(__mult_address_cache__[m_node_idx]);
        auto s = reinterpret_cast<half8 *>(__intermediate_address_cache__[s_node_idx].m_S);
        constexpr auto DIM_SCALE = sizeof(half8) / sizeof(half);

        // leading dimensions in half8: lds of S (N halves), ldm of the leaf products (M halves)
        // complexity = 4
        const auto lds = global_lds;
        const auto ldm = M / DIM_SCALE;
        auto S = s + yy * lds + xx;

        // M rows of S (offset of C21) and one M x M product (distance m_i -> m_{i+1})
        // complexity = 2
        const auto s_stride = M * lds;
        const auto m_stride = M * ldm;

        // units: pointers are T *, column offsets in T (DIM_SCALE halves), row offsets in
        // multiples of the leading dimension (lds for S, ldm for the products)
        {
            // destination: the parent's 2M x 2M product in the top 2M rows of its S area.
            // ldm == M / DIM_SCALE is also M halves, i.e. the column offset of C12 / C22
            // optimize
            // complexity = 4
            auto C11 = S;
            auto C12 = S + ldm;
            auto C21 = S + s_stride;
            auto C22 = S + s_stride + ldm;

            // sources: element (yy, xx) of each of the 7 products, stacked m_stride apart

            // complexity = 9
            const auto m1 = m + yy * ldm + xx;
            const auto m2 = m1 + m_stride;
            const auto m3 = m2 + m_stride;
            const auto m4 = m3 + m_stride;
            const auto m5 = m4 + m_stride;
            const auto m6 = m5 + m_stride;
            const auto m7 = m6 + m_stride;

            const auto m1x = cache ? fetch_no_cache(m1) : fetch_bypass_L2(m1);
            const auto m2x = cache ? fetch_no_cache(m2) : fetch_bypass_L2(m2);
            const auto m3x = cache ? fetch_no_cache(m3) : fetch_bypass_L2(m3);
            const auto m4x = cache ? fetch_no_cache(m4) : fetch_bypass_L2(m4);
            const auto m5x = cache ? fetch_no_cache(m5) : fetch_bypass_L2(m5);
            const auto m6x = cache ? fetch_no_cache(m6) : fetch_bypass_L2(m6);
            const auto m7x = cache ? fetch_no_cache(m7) : fetch_bypass_L2(m7);

            // C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6
            // complexity(op) = 4 * 3
            add_and_store<cache>(m1x, m2x, C11);
            sub_and_store<cache>(m5x, m7x, C12);
            add_and_store<cache>(m3x, m6x, C21);

            // C22 = m5 + m6 - m2 - m4 per half2 lane
            half2 result[4];
            {
                // complexity = 12
                result[0] = *reinterpret_cast<const half2 *>(&m5x.x)
                            + *reinterpret_cast<const half2 *>(&m6x.x)
                            - *reinterpret_cast<const half2 *>(&m2x.x)
                            - *reinterpret_cast<const half2 *>(&m4x.x);

                result[1] = *reinterpret_cast<const half2 *>(&m5x.y)
                            + *reinterpret_cast<const half2 *>(&m6x.y)
                            - *reinterpret_cast<const half2 *>(&m2x.y)
                            - *reinterpret_cast<const half2 *>(&m4x.y);

                result[2] = *reinterpret_cast<const half2 *>(&m5x.z)
                            + *reinterpret_cast<const half2 *>(&m6x.z)
                            - *reinterpret_cast<const half2 *>(&m2x.z)
                            - *reinterpret_cast<const half2 *>(&m4x.z);

                result[3] = *reinterpret_cast<const half2 *>(&m5x.w)
                            + *reinterpret_cast<const half2 *>(&m6x.w)
                            - *reinterpret_cast<const half2 *>(&m2x.w)
                            - *reinterpret_cast<const half2 *>(&m4x.w);
            }
            const auto &value = *reinterpret_cast<half8 *>(&result[0]);
            cache ? store_cache_L2(C22, value) : store_no_cache(C22, value);
        }
    }

    /**
     * @brief Combine the products of the 7 children (level l) of a node of level l - 1.
     * @param idx     destination node in __intermediate_address_cache__; its product (2M x 2M)
     *                goes into the top 2M rows of its S area
     * @param src_idx first of its 7 children (BFS order: idx's children are src_idx .. src_idx
     *                + 6, in the order m1 .. m7); each child's M x M product is in the top M
     *                rows of the child's S area, written by the previous step
     * @param lds     leading dimension of all S areas in T units (N / DIM_SCALE)
     * @param M       child size in halves (N >> l)
     * @param xx, yy  column (in T) and row inside the child product
     * @details Same formulas as leaf_block_reconstruct(); only the sources differ (seven
     * S areas instead of one continuous product buffer).
     */
    static __device__ __forceinline__ void intermediate_block_reconstruct(unsigned int idx,
                                                                          unsigned int src_idx,
                                                                          unsigned int lds,
                                                                          unsigned int M,
                                                                          unsigned int xx,
                                                                          unsigned int yy)
    {
        constexpr auto DIM_SCALE = sizeof(half8) / sizeof(half);
        // ldm: M halves in half8 = column offset of C12 / C22; s_stride: M rows of S
        const auto ldm = M / DIM_SCALE;
        const auto s_stride = M * lds;

        // the 7 child products m1 .. m7 at (yy, xx); same offset in every S area (same lds)
        half8 data[MULTS_COUNT];
        const auto offset_x = yy * lds + xx;

        // fetch data
        {
            // C11 C12
            // C21 C22

#pragma unroll
            // child i's product: top of its S area. __ldcg if cache
            // (else __ldcs). The 7 loads are independent and can all be in flight at once
            for (auto i = 0; i < MULTS_COUNT; ++i) { // copy async
                auto C = reinterpret_cast<const half8 *>(
                             __intermediate_address_cache__[src_idx + i].m_S)
                         + offset_x;
                const auto mx = cache ? fetch_cache_L2(C) : fetch_no_cache(C);
                data[i] = mx;
            }
        }

        {
            auto S = reinterpret_cast<half8 *>(__intermediate_address_cache__[idx + 0].m_S)
                     + offset_x;

            // optimize
            auto C11 = S;
            auto C12 = S + ldm;
            auto C21 = S + s_stride;
            auto C22 = S + s_stride + ldm;

            const auto &m1x = data[0];
            const auto &m2x = data[1];
            const auto &m3x = data[2];
            const auto &m4x = data[3];
            const auto &m5x = data[4];
            const auto &m6x = data[5];
            const auto &m7x = data[6];

            // C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6
            add_and_store<cache>(m1x, m2x, C11);
            sub_and_store<cache>(m5x, m7x, C12);
            add_and_store<cache>(m3x, m6x, C21);

            // C22 = m5 + m6 - m2 - m4 per half2 lane
            half2 result[4];
            {
                result[0] = *reinterpret_cast<const half2 *>(&m5x.x)
                            + *reinterpret_cast<const half2 *>(&m6x.x)
                            - *reinterpret_cast<const half2 *>(&m2x.x)
                            - *reinterpret_cast<const half2 *>(&m4x.x);

                result[1] = *reinterpret_cast<const half2 *>(&m5x.y)
                            + *reinterpret_cast<const half2 *>(&m6x.y)
                            - *reinterpret_cast<const half2 *>(&m2x.y)
                            - *reinterpret_cast<const half2 *>(&m4x.y);

                result[2] = *reinterpret_cast<const half2 *>(&m5x.z)
                            + *reinterpret_cast<const half2 *>(&m6x.z)
                            - *reinterpret_cast<const half2 *>(&m2x.z)
                            - *reinterpret_cast<const half2 *>(&m4x.z);

                result[3] = *reinterpret_cast<const half2 *>(&m5x.w)
                            + *reinterpret_cast<const half2 *>(&m6x.w)
                            - *reinterpret_cast<const half2 *>(&m2x.w)
                            - *reinterpret_cast<const half2 *>(&m4x.w);
            }
            const auto &value = *reinterpret_cast<half8 *>(&result[0]);
            cache ? store_cache_L2(C22, value) : store_no_cache(C22, value);
        }
    }
};
