/**
 * @file reference.hpp
 * @brief CPU reference for the GEMM pipeline, wired to MKL and the half-precision helpers
 * @author GEMM Project
 * @date 2025
 *
 * Thin compatibility layer over reference_core.hpp (see there for the math). Keeps the
 * historical global API used by the GPU tests:
 *
 *   encode(T, A), decode(T, C)              alternative-basis transforms (in place)
 *   mult_opt(T, A, B)                       A <- A * B in the alternative basis (MKL leaves)
 *   calculate_layers(R, AA, BB, nodes)      layered reference from half-packed inputs
 *
 * All functions are inline, so the header can be included from several translation units.
 * Used by the GPU tests (tst/precision.cpp, tst/unit_tests.cpp) to check the device results
 * of every layer; the pure CPU tests (tst/reference.cpp) use reference_core.hpp directly.
 *
 * Unlike reference_core.hpp this header pulls in common.hpp (and through hmat.hpp the CUDA
 * runtime and Qt) and its leaf GEMM needs MKL (cv::sgemm in common.cpp), so it is only usable
 * in the full GPU build.
 */
#pragma once

#include <reference_core.hpp>
#include <common.hpp>

// Export the transforms and the node type to the global namespace for the older test code.
using reference::decode;
using reference::decode_inverse;
using reference::encode;
using reference::encode_inverse;
using reference::test_node_t;

/// Leaf multiplication through MKL (cv::sgemm of common.hpp: cblas_sgemm, square CV_32FC1 blocks).
/// The wrappers below always pass float; the double tests use reference_core.hpp directly.
inline void mkl_leaf_gemm(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
{
    CV_Assert(A.type() == CV_32FC1);
    cv::sgemm(A, B, C);
}

/**
 * @brief A <- A * B in the alternative basis (both already encoded, result stays encoded).
 * @param T leaf size (power of two, T <= N); recursion stops at T x T blocks
 * @param A left operand, N x N CV_32FC1, overwritten with the encoded product
 * @param B right operand, same size and type
 * @note B is no longer modified (the old implementation used it as scratch).
 */
inline void mult_opt(int T, cv::Mat &A, const cv::Mat &B)
{
    reference::mult_opt(T, A, B, mkl_leaf_gemm);
}

/**
 * @brief Layered reference for GPU comparisons.
 * @param R  number of recursions (leaf size N >> R)
 * @param AA left operand in half-packed storage (N x N/2 CV_32FC1 holding N x N halves),
 *           already encoded if the GPU side encodes
 * @param BB right operand, same layout
 * @param nodes output: nodes[l] holds 7^l nodes in float; nodes[0][0].m_C is the product
 *              (still encoded: apply decode() to get A * B). See reference::calculate_layers
 *              for the node ordering (child 7 * j + i is product m(i+1) of node j, as in
 *              the GPU buffers described in bfs_caps.hpp).
 * @note The half inputs are widened to float first, so the reference does its sums and leaf
 *       products in float; differences to the GPU therefore include fp16 rounding on the GPU side.
 */
inline void calculate_layers(int R,
                             const cv::Mat &AA,
                             const cv::Mat &BB,
                             std::map<int, std::vector<test_node_t>> &nodes)
{
    reference::calculate_layers(R, to_float(AA), to_float(BB), nodes, mkl_leaf_gemm);
}
