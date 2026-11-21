/**
 * @file dfs_caps.hpp
 * @brief CAPS with a depth-first top level and products added straight into C (class DFSCaps)
 * @author GEMM Project
 * @date 2026
 *
 * The breadth-first CAPS of class BFS keeps the sums of every level, all 7^R leaf products and the
 * reconstructed products of every node at once (N = 65536, R = 2: 82 GB against 25.8 GB of a
 * plain GEMM). DFSCaps computes the same 7^R leaf products in the same alternative basis, but
 *   1. ENCODE   A and B are transformed in place, all R levels, once (as BFS);
 *   2. for each of the 7 products i of the root (depth first):
 *        form step: its two operands (an encoded quadrant, or the sum of two) are formed into dense
 *              (N/2) x (N/2) buffers, then the sums S1 .. S6 of every internal node of its
 *              R - 1 level sub-tree (no encode: the operands are already encoded at every level,
 *              the transform being linear and recursive);
 *        GEMM: its 7^(R-1) leaf products run as one batched wgmma GEMM whose epilogue adds every
 *              product, with its sign, straight into the blocks of C that the decoding matrix of
 *              the alternative basis (W x W x ..., one W per level) names
 *              (wgmma_gemm_t::gemm_reduce());
 *   3. DECODE   C is transformed back, all R levels, once.
 * There are no product buffers and no reconstruct pass.
 * W (src/reconstruct.cuh): C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6, C22 = m5 + m6 - m2 - m4.
 *
 * Modes (initialize()):
 *   serial      one buffer set, one stream: form step i, GEMM i, form step i + 1, ...
 *   overlapped  two buffer sets (A / B, alternating) and two streams: form step i + 1 runs during
 *               GEMM i. The persistent GEMM fills every SM, so the form step either gets SMs left
 *               free (set_form_sms(), the GEMM's persistent grid is capped) or runs at a higher
 *               stream priority (set_form_priority(), the graph keeps the node priorities).
 * The whole run is captured once into a CUDA graph (on the first run() and again when the
 * accumulator changes); run() then is one cudaGraphLaunch().
 *
 * Memory: A, B and C (3 N^2 halves) + per buffer set the two operands (2 (N/2)^2) and the sums
 * of the sub-tree (sums_rows() rows of N/2 halves); N = 65536, R = 2: 33 GB serial, 41 GB
 * overlapped. Needs the wgmma leaf GEMM (Hopper) and R >= 2.
 */

#pragma once

#include <graph.hpp>
#include <wgmma_gemm.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class DFSCaps
{
public:
    using accumulator_t = wgmma_gemm_t::accumulator_t;
    enum class mode_t { serial, overlapped };

    /// Device bytes of a run (A and B of the caller included, as BFSCaps::required_memory())
    static size_t required_memory(int N, int R, mode_t mode);
    /// True when required_memory() + 2 GiB reserve fits the device
    static bool fits_device_memory(int N, int R, mode_t mode, std::string *why);

    ~DFSCaps();

    /**
     * @brief Prepare C = A * B: allocate C and the buffer set(s), build the sub-tree, the target
     *        lists and the leaf plans
     * @param A, B N x N halves as N x N / 2 CV_32FC1 continuous GpuMats; overwritten by run().
     *        N = k * 2^R with the leaf size k a multiple of 256 (the wgmma leaf GEMM); N need not
     *        be a power of two (e.g. 49152 = 12288 * 2^2)
     * @param R recursions, 2 .. MAX_RECURSIONS
     * @return false (nothing allocated) when N is not of that form, it does not fit or wgmma is
     *         not available
     */
    bool initialize(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, int R, cv::cuda::Stream &stream,
                    std::string *why = nullptr, mode_t mode = mode_t::serial);

    /// overlapped: SMs left free for the form step while a GEMM runs (the GEMM's persistent grid is
    /// capped at SMs - n); 0: none; AUTO_FORM_SMS (the default): auto_form_sms(N). Takes effect at
    /// the next capture.
    static constexpr int AUTO_FORM_SMS = -1;
    void set_form_sms(int n) { m_form_sms = n; m_graph.reset(); }
    /// The automatic choice, from the H100 dfs-check (R = 2, % vs BFS CAPS): N <= 16384 8 SMs
    /// (+4.9 % serial -> -0.3 %), 32768 4 SMs (-0.4 % -> -2.0 %; 16 SMs: +4.2 %), larger N none
    /// (the GEMM dominates, every SM is worth more than the hidden form step)
    static int auto_form_sms(int N) { return N <= 16384 ? 8 : N <= 32768 ? 4 : 0; }
    /// overlapped: the form stream gets the highest priority (default false: alone it did not help on
    /// the H100, the persistent GEMM holds every SM until it ends). Before the next initialize().
    void set_form_priority(bool high) { m_form_priority = high; }

    /// The whole multiplication as one CUDA graph launch (captured on the first call); the result
    /// is then get_result()
    void run(cv::cuda::Stream &stream, accumulator_t accumulator);

    /// C (N x N halves as N x N / 2 CV_32FC1), valid after run()
    const cv::cuda::GpuMat &get_result() const { return m_C; }

    /// The leaf GEMM, for its settings and last_launch()
    wgmma_gemm_t &leaf() { return m_wgmma; }

    int get_dimension() const { return m_N; }
    int get_recursions() const { return m_R; }
    mode_t mode() const { return m_mode; }

private:
    /// One internal node of the sub-tree: operands and its six sums, all with pitch N / 2 halves
    struct node_t
    {
        half *L, *R, *S;
    };
    /// A buffer set: the sub-problem's operands, the sums of its sub-tree, the leaf plan
    struct set_t
    {
        cv::cuda::GpuMat subA, subB, S;
        std::vector<std::shared_ptr<void>> levels; ///< device node_t arrays, one per sub-tree level
        std::vector<int> counts;                   ///< nodes per level
        wgmma_gemm_t::plan_t plan;                 ///< tensor maps of the leaf operands
    };
    /// Rows of N / 2 halves of one set's sum buffer (sub-tree of size H = N / 2, R - 1 levels)
    static size_t sums_rows(int H, int levels);
    bool build_set(set_t &set, std::string &why);
    /// The work of run() on the streams (captured into the graph)
    void enqueue(cv::cuda::Stream &stream, accumulator_t accumulator);
    /// Root product i into @p set on stream @p s: its two operands formed into the set's dense
    /// buffers (form_operand_kernel), then the sums of its R - 1 level sub-tree (sums_kernel)
    void form_subtree(set_t &set, int i, cudaStream_t s);
    void release();

    int m_N = 0;
    int m_R = 0;
    int m_T = 0;      ///< targets per leaf product (2^R)
    int m_leaves = 0; ///< leaf products per sub-problem, 7^(R-1)
    mode_t m_mode = mode_t::serial;
    int m_form_sms = AUTO_FORM_SMS;
    bool m_form_priority = false;
    cv::cuda::GpuMat m_A, m_B; ///< headers of the caller's operands
    cv::cuda::GpuMat m_C;      ///< the result, N x N halves
    std::vector<set_t> m_sets; ///< 1 (serial) or 2 (overlapped)
    wgmma_gemm_t m_wgmma;
    std::shared_ptr<void> m_targets; ///< device: 7 x m_leaves x m_T pointers into C (bit 0: subtract)
    cudaStream_t m_form_stream = nullptr;           ///< overlapped: the form stream
    std::vector<cudaEvent_t> m_events;              ///< overlapped: form step done / GEMM done per product, + fork
    std::unique_ptr<CaptureGraph> m_graph;          ///< run()'s graph, captured for m_graph_accumulator
    accumulator_t m_graph_accumulator = accumulator_t::half;
};
