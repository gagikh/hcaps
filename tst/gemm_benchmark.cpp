/**
 * @file gemm_benchmark.cpp
 * @brief GEMM benchmark suite: performance and precision of cuBLAS and CAPS + cuBLAS
 *
 * Measures only: the results go to CSV files in plots/data (results.hpp), and
 * python py/plots.py draws the graphs into plots/.
 *
 *     ./bin/test_gemm_benchmark [--max-dim=...] [--max-recursions=...] [--results=<dir>]
 *     python py/plots.py
 */

#include <common.hpp>
#include <perf.hpp>
#include <map>
#include "test_utils.hpp"

#include "performance_tests.hpp"
#include "results.hpp"

#include <bfs_caps.hpp>
#include <dfs_caps.hpp>
#include <core.hpp>
#include <hmat.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <cublas_gemm.hpp>
#include <wgmma_gemm.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublasLt.h> // "CUBLAS best": the fastest candidate of cuBLASLt's heuristic
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <QCoreApplication>
#include <iosfwd>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <string>
#include <utility>
#include <filesystem>
#include <fstream>
#include <sstream>

/**
 * @namespace GEMM
 * @brief Main namespace for GEMM project functionality
 * 
 * Contains all classes, functions, and utilities for the GEMM implementation
 * including benchmarking and testing.
 */
namespace GEMM {

/// Accumulator of every timed GEMM (--accumulator): f16 (CUBLAS_COMPUTE_16F, half alpha / beta) or
/// f32 (CUBLAS_COMPUTE_32F, float alpha / beta), the same for cuBLAS, our kernel and the CAPS leaves
static wgmma_gemm_t::accumulator_t accumulator = wgmma_gemm_t::accumulator_t::half;

static bool f32_accumulation()
{
    return accumulator == wgmma_gemm_t::accumulator_t::single;
}

static cublasComputeType_t compute_type()
{
    return f32_accumulation() ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
}

/// alpha = 1 and beta = 0 in the scale type of the compute type: half for CUBLAS_COMPUTE_16F, float
/// for CUBLAS_COMPUTE_32F (cuBLAS reads 4 bytes there; a half 1.0 read as a float is ~0)
struct unit_scale_t
{
    __half m_one_h = __float2half(1.0f);
    __half m_zero_h = __float2half(0.0f);
    float m_one_f = 1.0f;
    float m_zero_f = 0.0f;
    const void *one() const { return f32_accumulation() ? static_cast<const void *>(&m_one_f) : &m_one_h; }
    const void *zero() const { return f32_accumulation() ? static_cast<const void *>(&m_zero_f) : &m_zero_h; }
};

/// cuBLAS's default GEMM of N x N half matrices (N x N / 2 CV_32FC1, row-major, as
/// cublas_gemm_t::gemm()) with f32 accumulation: cublasGemmEx, CUBLAS_COMPUTE_32F, computing the
/// column-major C^T = B^T A^T
static void default_gemm_f32(cublas_gemm_t &cgemm, cv::cuda::Stream &stream, const cv::cuda::GpuMat &A,
                             const cv::cuda::GpuMat &B, cv::cuda::GpuMat &C)
{
    cgemm.set_stream(stream);
    const auto N = A.rows;
    const auto ld = int(A.step / sizeof(half));
    const auto ldc = int(C.step / sizeof(half));
    const float one = 1.0f;
    const float zero = 0.0f;
    const auto status = cublasGemmEx(cgemm.handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, B.data, CUDA_R_16F, ld,
                                     A.data, CUDA_R_16F, ld, &zero, C.data, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx (f32 accumulation) failed: " + std::to_string(int(status)));
    }
}

/// The GEMM that the "cublas" series times: cublasHgemm (f16) or cublasGemmEx (f32)
static void default_gemm(cublas_gemm_t &cgemm, cv::cuda::Stream &stream, const cv::cuda::GpuMat &A,
                         const cv::cuda::GpuMat &B, cv::cuda::GpuMat &C)
{
    if (f32_accumulation()) {
        default_gemm_f32(cgemm, stream, A, B, C);
    } else {
        cgemm.gemm(A, B, C);
    }
}

/**
 * @brief cuBLAS at its best for one N x N half GEMM (compute 16F, like cublasHgemm)
 * @details cublasHgemm runs the first candidate of cuBLASLt's heuristic, which is not always the
 * fastest (H100, N = 32768: rank 0 682 TFLOPS, rank 2 824 TFLOPS; test_diagnostics cublas-algos).
 * The constructor asks the heuristic for up to CANDIDATES candidates for these operands, runs each
 * once (failures are skipped) and keeps the fastest; run() then launches that one. Selection at the
 * clock the benchmark measures with: first GEMM::warm_up_ms of warm-up (the H100 reaches its power
 * cap ~1 s into a load and then runs ~10 % slower), then ROUNDS rounds over all candidates, each
 * candidate timed with the same number of calls per round (enough for ~TARGET_MS, at least 1),
 * scored by its median round. The first version timed each candidate once (3 calls, no warm-up):
 * the early candidates ran on a cool GPU, and at N = 16384 "cuBLAS best" was 4 % slower than
 * cublasHgemm's own choice. Column-major, ld = the rows' pitch: the operands are square and
 * the result is not used, so only the speed matters. 64 MB workspace (cuBLAS's upper default).
 */
class cublas_best_t
{
public:
    static constexpr int CANDIDATES = 32;
    static constexpr int ROUNDS = 3;          ///< timing rounds over all candidates (median: the middle one)
    static constexpr float TARGET_MS = 2.0f;  ///< about this long per candidate and round

    /// One N x N product (the rows' pitch as ld)
    cublas_best_t(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::GpuMat &C, cv::cuda::Stream &stream)
        : cublas_best_t(A.data, B.data, C.data, uint64_t(A.rows), int64_t(A.step / sizeof(half)), 1, 0,
                        (cudaStream_t) stream.cudaPtr())
    {
        CV_Assert(A.step == B.step && A.step == C.step && A.rows == B.rows && A.rows == C.rows);
    }

    /// A batch of `count` N x N products (cuBLASLt layout attribute BATCH_COUNT when count > 1).
    /// MODE = CUBLASLT_BATCH_MODE_STRIDED: operand i at A + i * stride (in halfs), the same for B and C
    /// (STRIDED_BATCH_OFFSET). MODE = CUBLASLT_BATCH_MODE_POINTER_ARRAY: A, B and C are device arrays
    /// of `count` operand addresses, as CAPS addresses its scattered leaves; stride is not used.
    cublas_best_t(const void *A, const void *B, void *C, uint64_t N, int64_t ld, int count, int64_t stride,
                  cudaStream_t stream, cublasLtBatchMode_t mode = CUBLASLT_BATCH_MODE_STRIDED)
        : m_A(A), m_B(B), m_C(C), m_stream(stream)
    {
        const std::string what = "cuBLAS best" + std::string(CUBLASLT_BATCH_MODE_POINTER_ARRAY == mode ? " pointers" : "")
                                 + ", N=" + std::to_string(N) + (count > 1 ? " x " + std::to_string(count) : std::string());
        if (cublasLtCreate(&m_lt) != CUBLAS_STATUS_SUCCESS || cudaMalloc(&m_workspace, WORKSPACE) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        cublasLtMatmulDescCreate(&m_op, compute_type(), f32_accumulation() ? CUDA_R_32F : CUDA_R_16F);
        cublasLtMatrixLayoutCreate(&m_layout, CUDA_R_16F, N, N, ld);
        if (count > 1) {
            const int32_t batch = count;
            cublasLtMatrixLayoutSetAttribute(m_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
            if (CUBLASLT_BATCH_MODE_POINTER_ARRAY == mode) {
                const uint32_t batch_mode = mode;
                cublasLtMatrixLayoutSetAttribute(m_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode));
            } else {
                cublasLtMatrixLayoutSetAttribute(m_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
            }
        }
        cublasLtMatmulPreference_t preference = nullptr;
        cublasLtMatmulPreferenceCreate(&preference);
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WORKSPACE,
                                             sizeof(WORKSPACE));
        std::vector<cublasLtMatmulHeuristicResult_t> results(CANDIDATES);
        int found = 0;
        cublasLtMatmulAlgoGetHeuristic(m_lt, m_op, m_layout, m_layout, m_layout, m_layout, preference, CANDIDATES,
                                       results.data(), &found);
        cublasLtMatmulPreferenceDestroy(preference);
        if (0 == found) {
            std::cerr << what << ": the heuristic returned no candidate" << std::endl;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // ms of `calls` back-to-back calls of algo (CUDA events)
        const auto time_calls = [&](const cublasLtMatmulAlgo_t &algo, int calls) {
            cudaEventRecord(start, m_stream);
            for (int i = 0; i < calls; ++i) {
                launch(algo);
            }
            cudaEventRecord(stop, m_stream);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            return ms;
        };
        // the candidates that run here (one call each; failures skipped), with the time of that call
        std::vector<int> ranks;
        float slowest_call_ms = 0;
        for (int r = 0; r < found; ++r) {
            const auto &algo = results[size_t(r)].algo;
            const auto status = launch(algo);
            const auto synced = cudaStreamSynchronize(m_stream);
            if (status != CUBLAS_STATUS_SUCCESS || synced != cudaSuccess) {
                // this candidate does not run here: skip it. A launch error clears; a fault in the
                // kernel (e.g. an illegal instruction) is sticky and leaves the context unusable:
                // then say which candidate it was and stop (batched_costs at L = 256 x 343 died so)
                std::cerr << what << ": candidate rank " << r << " failed (status " << int(status) << ", "
                          << cudaGetErrorString(synced) << ")" << std::endl;
                cudaGetLastError();
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    throw std::runtime_error("cuBLAS best: candidate rank " + std::to_string(r)
                                             + " left the CUDA context unusable");
                }
                continue;
            }
            ranks.push_back(r);
            slowest_call_ms = std::max(slowest_call_ms, time_calls(algo, 1));
        }
        if (!ranks.empty()) {
            // warm-up to the steady (power-capped) clock with the first candidate, as run_test() does
            const auto &first = results[size_t(ranks.front())].algo;
            if (GEMM::warm_up_ms > 0) {
                const auto begin = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - begin < std::chrono::milliseconds(GEMM::warm_up_ms)) {
                    time_calls(first, 1);
                }
            }
            // the same number of calls for every candidate: ~TARGET_MS for the slowest one
            const int calls = std::max(1, std::min(50, int(TARGET_MS / std::max(slowest_call_ms, 1e-3f))));
            // ROUNDS rounds, candidate after candidate, so that every candidate sees the same clock drift
            std::vector<std::vector<float>> times(ranks.size());
            for (int round = 0; round < ROUNDS; ++round) {
                for (size_t c = 0; c < ranks.size(); ++c) {
                    times[c].push_back(time_calls(results[size_t(ranks[c])].algo, calls) / float(calls));
                }
            }
            auto best_ms = -1.0f, first_ms = -1.0f;
            for (size_t c = 0; c < ranks.size(); ++c) {
                std::sort(times[c].begin(), times[c].end());
                const auto median = times[c][times[c].size() / 2];
                if (0 == ranks[c]) {
                    first_ms = median;
                }
                if (best_ms < 0 || median < best_ms) {
                    best_ms = median;
                    m_algo = results[size_t(ranks[c])].algo;
                    m_rank = ranks[c];
                    m_ok = true;
                }
            }
            std::cerr << what << ": rank " << m_rank << " of " << found << " candidates, "
                      << ROUNDS << " rounds of " << calls << " calls, " << describe(m_algo);
            if (m_ok && first_ms > 0) {
                std::cerr << ", " << 100.0 * (best_ms - first_ms) / first_ms << " % vs rank 0 (the cublasHgemm choice)";
            }
            std::cerr << std::endl;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~cublas_best_t()
    {
        if (m_layout) {
            cublasLtMatrixLayoutDestroy(m_layout);
        }
        if (m_op) {
            cublasLtMatmulDescDestroy(m_op);
        }
        if (m_lt) {
            cublasLtDestroy(m_lt);
        }
        cudaFree(m_workspace);
    }

    cublas_best_t(const cublas_best_t &) = delete;
    cublas_best_t &operator=(const cublas_best_t &) = delete;

    /// a candidate ran and was chosen
    bool ok() const { return m_ok; }
    /// C = A * B with the chosen kernel (asynchronous on the constructor's stream)
    void run() { launch(m_algo); }

private:
    static constexpr size_t WORKSPACE = size_t(64) << 20;

    /// The configuration of a cuBLASLt candidate as text: algo id, tile (cublasLtMatmulTile_t,
    /// named for the classic shapes, else its number), stages (cublasLtMatmulStages_t number),
    /// cluster (cublasLtClusterShape_t, named for the common shapes), split-K
    static std::string describe(const cublasLtMatmulAlgo_t &algo)
    {
        int id = -1;
        int tile = -1;
        int stages = -1;
        int splitk = -1;
        int cluster = -1;
        size_t written = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &written);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &written);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &written);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster),
                                             &written);
        const auto tile_name = [](int t) -> std::string {
            switch (t) {
            case CUBLASLT_MATMUL_TILE_64x64: return "64x64";
            case CUBLASLT_MATMUL_TILE_128x32: return "128x32";
            case CUBLASLT_MATMUL_TILE_64x128: return "64x128";
            case CUBLASLT_MATMUL_TILE_128x64: return "128x64";
            case CUBLASLT_MATMUL_TILE_64x256: return "64x256";
            case CUBLASLT_MATMUL_TILE_128x128: return "128x128";
            case CUBLASLT_MATMUL_TILE_256x64: return "256x64";
            case CUBLASLT_MATMUL_TILE_128x256: return "128x256";
            case CUBLASLT_MATMUL_TILE_256x128: return "256x128";
            default: return "#" + std::to_string(t);
            }
        };
        const auto cluster_name = [](int c) -> std::string {
            switch (c) {
            case CUBLASLT_CLUSTER_SHAPE_AUTO: return "auto";
            case CUBLASLT_CLUSTER_SHAPE_1x1x1: return "1x1";
            case CUBLASLT_CLUSTER_SHAPE_2x1x1: return "2x1";
            case CUBLASLT_CLUSTER_SHAPE_1x2x1: return "1x2";
            case CUBLASLT_CLUSTER_SHAPE_2x2x1: return "2x2";
            case CUBLASLT_CLUSTER_SHAPE_4x1x1: return "4x1";
            case CUBLASLT_CLUSTER_SHAPE_1x4x1: return "1x4";
            default: return "#" + std::to_string(c);
            }
        };
        return "algo " + std::to_string(id) + ", tile " + tile_name(tile) + ", stages #" + std::to_string(stages)
               + ", cluster " + cluster_name(cluster) + ", split-K " + std::to_string(splitk);
    }

    cublasStatus_t launch(const cublasLtMatmulAlgo_t &algo)
    {
        return cublasLtMatmul(m_lt, m_op, m_scale.one(), m_A, m_layout, m_B, m_layout, m_scale.zero(), m_C, m_layout, m_C, m_layout,
                              &algo, m_workspace, WORKSPACE, m_stream);
    }

    const void *m_A;
    const void *m_B;
    void *m_C;
    cudaStream_t m_stream;
    cublasLtHandle_t m_lt = nullptr;
    cublasLtMatmulDesc_t m_op = nullptr;
    cublasLtMatrixLayout_t m_layout = nullptr; ///< A, B and C: the same N x N layout
    void *m_workspace = nullptr;
    cublasLtMatmulAlgo_t m_algo{};
    unit_scale_t m_scale; ///< alpha / beta in the scale type of compute_type()
    int m_rank = -1;
    bool m_ok = false;
};

/// --dfs-caps: also time DFSCaps (depth-first top level, leaf products reduce-added into C; wgmma
/// builds, R >= 2) next to BFS in relative_costs() and the precision runs
static bool dfs_caps = true;
/// --sizes: the N of relative_costs() (comma separated, any k * 2^R); empty: the powers of two
/// from 256 up to --max-dim
static std::vector<int> sizes;

static std::vector<int> benchmark_sizes()
{
    if (!sizes.empty()) {
        return sizes;
    }
    std::vector<int> out;
    for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
        out.push_back(N);
    }
    return out;
}

#if (LEAF_GEMM_WGMMA)
/// Our GEMM on one N x N product C = A * B (R = 0, no recursion). The one-entry pointer arrays
/// and the tensor maps are made here, outside any timed call, so nothing synchronizes inside a
/// measurement. The arrays are filled on the host and copied (a cudaMemcpy): device_vector(n,
/// value) would run a fill kernel, which a file compiled by the host compiler cannot instantiate.
struct own_gemm_t
{
    own_gemm_t(cv::cuda::GpuMat &A, cv::cuda::GpuMat &B, cv::cuda::GpuMat &C, cv::cuda::Stream &stream)
        : m_A(one(A)), m_B(one(B)), m_C(one(C)), m_ldc(int(C.step / sizeof(half))),
          m_plan(wgmma_gemm_t::make_plan(m_A, int(A.step / sizeof(half)), m_B, int(B.step / sizeof(half)), A.rows))
    {
        m_gemm.set_stream(stream);
    }
    void run(wgmma_gemm_t::accumulator_t accumulator) { m_gemm.gemm(m_plan, m_A, m_C, m_ldc, accumulator); }

private:
    static thrust::device_vector<half *> one(cv::cuda::GpuMat &M)
    {
        return thrust::device_vector<half *>(thrust::host_vector<half *>(1, reinterpret_cast<half *>(M.data)));
    }
    const thrust::device_vector<half *> m_A;
    const thrust::device_vector<half *> m_B;
    thrust::device_vector<half *> m_C;
    int m_ldc;
    wgmma_gemm_t::plan_t m_plan;
    wgmma_gemm_t m_gemm;
};
#endif

/// Device memory in use (total - free) in GiB
static double device_used_gib()
{
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return double(total_bytes - free_bytes) / double(1ull << 30);
}

// Additional utility functions
static void relative_costs()
{
    cv::cuda::Stream stream;

    std::map<int, double> target_cost;
    std::map<int, double> own_cost; // our own GEMM (the leaf GEMM of this build) on N x N, R = 0
    std::map<int, double> best_cost; // cuBLAS's fastest heuristic candidate (cublas_best_t)
    std::map<int, std::map<int, double>> measured_cost;
    std::map<int, std::map<int, double>> dfs_cost;   // DFSCaps serial, R -> N -> cost
    std::map<int, std::map<int, double>> dfs_overlap_cost; // DFSCaps overlapped (automatic form SMs)
    std::map<std::string, std::map<int, double>> memory;   // series -> N -> device GiB while it runs
    {
        cublas_gemm_t cgemm(stream);
        for (const auto N : benchmark_sizes()) {
            if (!GEMM::inputs_fit(N)) {
                continue; // --sizes need not be ascending
            }
            // memory.csv: device memory in use while a series runs, minus what was in use before
            // the inputs of this N, so it counts A, B (and C for the cuBLAS series) and every
            // buffer the method holds
            const double base = device_used_gib();
            const auto record = [&](const std::string &name) { memory[name][N] = device_used_gib() - base; };
            cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1), B_dev(N, N / 2, CV_32FC1),
                C_dev(N, N / 2, CV_32FC1);
            {
                cuperf_t perf("cublas");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &stream) { default_gemm(cgemm, stream, A_dev, B_dev, C_dev); },
                    stream, measure);
                target_cost[N] = perf.get_avg();
                record("CUBLAS");
            }
            // "CUBLAS best": the same GEMM with cuBLAS's fastest candidate, chosen by timing (the
            // fair baseline; "CUBLAS" above is what a cublasHgemm call gets)
            try {
                cublas_best_t best(A_dev, B_dev, C_dev, stream);
                if (best.ok()) {
                    cuperf_t perf("cublas_best");
                    GEMM::run_test(perf, [&](cv::cuda::Stream &) { best.run(); }, stream, measure);
                    best_cost[N] = perf.get_avg();
                    record("CUBLAS best");
                }
            } catch (const std::exception &e) {
                std::cerr << "cuBLAS best, N=" << N << ": skipped (" << e.what() << ")" << std::endl;
            }
#if (LEAF_GEMM_WGMMA)
            // our own GEMM on the whole N x N product (R = 0, no recursion), the kernel this build
            // uses for the leaves (CMake LEAF_GEMM), half accumulation like CAPS: the second
            // baseline, so that CAPS is also compared with its own leaf kernel, not only with
            // cuBLAS (own_gemm_t). A size the kernel cannot run (not a multiple of 256) is left out.
            try {
                if (wgmma_gemm_t::supports(N)) {
                    own_gemm_t own(A_dev, B_dev, C_dev, stream);
                    cuperf_t perf("own");
                    GEMM::run_test(perf, [&](cv::cuda::Stream &) { own.run(GEMM::accumulator); }, stream, measure);
                    own_cost[N] = perf.get_avg();
                    record("OWN wgmma");
                }
            } catch (const std::exception &e) {
                std::cerr << "own GEMM, N=" << N << ": skipped (" << e.what() << ")" << std::endl;
            }
#endif
            // C_dev is not needed by CAPS (it writes its own result): free it, BFSCaps::fits_device_memory()
            // counts only A and B of the caller, and at N = 65536 C_dev (8 GiB) made R = 2 run out of memory
            C_dev.release();
            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
#if (LEAF_GEMM_WGMMA)
                for (const auto mode : {DFSCaps::mode_t::serial, DFSCaps::mode_t::overlapped}) {
                    const bool overlapped = mode == DFSCaps::mode_t::overlapped;
                    const char *name = overlapped ? "DFSCaps overlapped" : "DFSCaps";
                    if (!(dfs_caps && R >= 2 && (N >> R) >= 256)) { // before is_valid(): it checks the BFS memory
                        break;
                    }
                    try {
                        DFSCaps dfs;
                        std::string why;
                        if (!dfs.initialize(A_dev, B_dev, R, stream, &why, mode)) {
                            std::cerr << "skip: " << name << " N=" << N << ", R=" << R << ": " << why << std::endl;
                            continue;
                        }
                        cuperf_t perf(overlapped ? "dfs_caps_overlapped" : "dfs_caps");
                        GEMM::run_test(
                            perf, [&](cv::cuda::Stream &stream) { dfs.run(stream, GEMM::accumulator); }, stream, measure);
                        (overlapped ? dfs_overlap_cost : dfs_cost)[R][N] = perf.get_avg();
                        record(std::string(name) + " R=" + std::to_string(R));
                    } catch (const std::exception &e) {
                        std::cerr << "skip: " << name << " N=" << N << ", R=" << R << ": " << e.what() << std::endl;
                    }
                }
#endif
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                // a missing kernel (no tuning_table.hpp entry for (N, R) when the loop counts are
                // template arguments) skips this (N, R) instead of aborting the whole benchmark
                try {
                    BFSCaps bfs;
                    if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                        continue;
                    }

                    cuperf_t perf("bfs");
                    GEMM::run_test(
                        perf,
                        [&](cv::cuda::Stream &stream) {
                            bfs.forward_pass(stream);
                            bfs.batched_leaf(stream, GEMM::accumulator);
                            bfs.backward_pass(stream);
                        },
                        stream, measure);
                    measured_cost[R][N] = perf.get_avg();
                    record("CAPS R=" + std::to_string(R));
                } catch (const std::exception &e) { // BFS released first: one instance at a time
                    std::cerr << "skip: CAPS N=" << N << ", R=" << R << ": " << e.what() << std::endl;
                    continue;
                }
            }
        }
    }
    // generate csv
    std::ofstream out;
    {
        std::stringstream ss;
        ss << "final-costs.csv";
        out.open(ss.str(), std::ios::out | std::ios::trunc);
    }

    out << "algo/dimension";
    std::map<std::string, std::map<int, double>> series;
    {
        for (const auto &it : target_cost) {
            out << ", " << it.first;
        }
        out << std::endl << "cublas";
        for (const auto &it : target_cost) {
            out << ", " << it.second;
        }
        series["CUBLAS"] = target_cost;
    }
    {
        for (const auto &it : measured_cost) {
            const auto &R = it.first;
            const auto &container = it.second;
            out << std::endl << "caps R=" << R;
            for (const auto &it : target_cost) {
                const auto &N = it.first;
                auto jt = container.find(N);
                if (container.end() == jt) {
                    out << ", x";
                } else {
                    series["CAPS R=" + std::to_string(R)][N] = jt->second;
                    out << ", " << jt->second;
                }
            }
        }
    }
    out << std::endl;
    out.close();
    for (const auto &it : dfs_cost) {
        series["DFSCaps R=" + std::to_string(it.first)] = it.second;
    }
    for (const auto &it : dfs_overlap_cost) {
        series["DFSCaps overlapped R=" + std::to_string(it.first)] = it.second;
    }
    if (!best_cost.empty()) { // py/plots.py also draws everything relative to it
        series["CUBLAS best"] = best_cost;
    }
    if (!own_cost.empty()) { // "OWN <kernel>": py/plots.py also draws everything relative to it
        series["OWN wgmma"] = own_cost;
    }

    // absolute times (or energies); py/plots.py draws them as "Costs" and their ratio to cuBLAS as
    // "Relative Costs" (the file keeps its name for the stored runs)
    write_series("relative_costs", "Costs", GEMM::units_text(GEMM::measure), series);
    write_series("memory", "Device Memory", "GiB", memory);
}

/// --batched: also run batched_costs() after relative_costs()
static bool batched = true;
/// --relative-costs: run relative_costs() (0: only the batched benchmark, e.g. to debug it)
static bool relative = true;
/// --batched-from-r: the first R of batched_costs() (e.g. 3: start at 343 leaves, without the
/// kernels of the smaller batches before it)
static int batched_from_r = 1;
/// --batched-own: time our wgmma in batched_costs() (0: cuBLAS only, to tell whether a crash
/// there comes from cuBLAS's kernels alone)
static bool batched_own = true;
/// --batched-keep-plans: keep every plan of our wgmma (its tensor maps in device memory) alive
/// until batched_costs() ends, instead of freeing it after its size. A test: the crash in the
/// cuBLAS kernel after ours (343 leaves) might come from a stale tensor-map cache entry of our
/// freed maps, whose memory cuBLAS then reuses; with the plans kept, that memory cannot be reused.
static bool batched_keep_plans = false;
/// --batched-cublas: time the cuBLAS batched GEMMs in batched_costs() (0: ours only; with
/// --batched-own=0 in a second process the table is complete, py/plots.py takes both files merged)
static bool batched_cublas = true;
/// --batched-direct: our wgmma with the direct store of C (set_staged_store(false)), a test
static bool batched_direct = false;
/// --batched-canary: after every size of batched_costs(), run our wgmma once on a fixed small batch
/// (49 x 256, f32 accumulation, integer data; plan and buffers alive for the whole run) and check
/// the CUDA error state and the result against its first run, bit by bit: the first size after
/// which it fails is the one that corrupted memory (the batched benchmark crashed in the kernel
/// after ours, our own kernel included)
static bool batched_canary = false;
/// --batched-only: only these (R, leaf) sizes of batched_costs(), "R:L,R:L,..." (empty: all), run
/// in the usual order (R, then L ascending); to bisect which size a later failure depends on
static std::string batched_only;

/**
 * @brief batched_costs: the leaf GEMM phase of CAPS on its own, our batched GEMM against cuBLAS's
 *        batched GEMMs (results file batched_costs.csv, py/plots.py: Relative_Costs_batched_R<r>_wgmma_vs_cuBLAS.png)
 * @details For R = 1 .. max_recursions and every leaf size L = 256, 512, ... with L * 2^R <
 * max_dimension: a batch of 7^R products of L x L halves (the leaves of CAPS at N = L * 2^R), one
 * buffer per operand kind (A, B, C, product i at i * L * L), timed with run_test() as relative_costs():
 *   "CUBLAS batched R=r"      cublasGemmStridedBatchedEx, compute 16F (what a user calls),
 *   "CUBLAS pointers R=r"     cublasGemmBatchedEx with device pointer arrays (how CAPS addresses
 *                             its scattered leaves),
 *   "CUBLAS best batched R=r" cuBLASLt's fastest heuristic candidate for the strided batch
 *                             (cublas_best_t: warm-up, interleaved rounds),
 *   "CUBLAS best pointers R=r" the same for the products given as device pointer arrays
 *                             (cuBLASLt CUBLASLT_BATCH_MODE_POINTER_ARRAY, the addressing of CAPS),
 *   "OWN wgmma R=r"           our wgmma_gemm_t, one launch for the batch, half accumulation (the
 *                             leaf GEMM of CAPS; LEAF_GEMM=wgmma builds only).
 * x = L. Operands filled with the byte 0x3C (every half ~1.06: finite). Sizes whose three buffers
 * do not fit into the free device memory minus 2 GiB are skipped.
 */
static void batched_costs()
{
    cv::cuda::Stream stream;
    const auto s = (cudaStream_t) stream.cudaPtr();
    // the project includes the legacy <cublas.h> (cublas_gemm.hpp), and cublas_v2.h must not be
    // included next to it: the handle API by its explicit _v2 names, as src/cublas_gemm.cpp does
    cublasHandle_t handle = nullptr;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "batched_costs: cublasCreate failed" << std::endl;
        return;
    }
    cublasSetStream_v2(handle, s);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    const unit_scale_t scale; // alpha / beta of the cuBLAS calls below (--accumulator)
    std::map<std::string, std::map<int, double>> series;
#if (LEAF_GEMM_WGMMA)
    std::vector<wgmma_gemm_t::plan_t> kept_plans; // --batched-keep-plans
    // --batched-canary (see there): its buffers, plan, reference result and check
    constexpr int CANARY_L = 256;
    constexpr int CANARY_COUNT = 49;
    const auto canary_n = size_t(CANARY_COUNT) * size_t(CANARY_L) * size_t(CANARY_L);
    half *canary = nullptr; // A, B, C of the canary batch, one after the other
    wgmma_gemm_t::plan_t canary_plan;
    thrust::device_vector<half *> canary_c;
    std::vector<uint16_t> canary_reference;
    wgmma_gemm_t canary_gemm;
    canary_gemm.set_stream(stream);
    if (batched_canary && wgmma_gemm_t::supports(CANARY_L)) {
        cudaMalloc(&canary, 3 * canary_n * sizeof(half));
        std::vector<uint16_t> host(2 * canary_n);
        uint32_t state = 777u;
        for (auto &h : host) {
            state = state * 1664525u + 1013904223u;
            const auto v = __float2half(float(int((state >> 16) % 4u) - 2)); // integers in [-2, 1]
            std::memcpy(&h, &v, sizeof(h));
        }
        cudaMemcpy(canary, host.data(), 2 * canary_n * sizeof(half), cudaMemcpyHostToDevice);
        const auto per = size_t(CANARY_L) * size_t(CANARY_L);
        const auto canary_products = size_t(CANARY_COUNT); // named: ha(size_t(...)) would declare a function
        thrust::host_vector<half *> ha(canary_products);
        thrust::host_vector<half *> hb(canary_products);
        thrust::host_vector<half *> hc(canary_products);
        for (size_t i = 0; i < canary_products; ++i) {
            ha[i] = canary + i * per;
            hb[i] = canary + canary_n + i * per;
            hc[i] = canary + 2 * canary_n + i * per;
        }
        canary_c = hc;
        canary_plan = wgmma_gemm_t::make_plan(ha.data(), CANARY_L, hb.data(), CANARY_L, CANARY_L, CANARY_COUNT, nullptr);
        canary_reference.resize(canary_n);
        canary_gemm.gemm(canary_plan, thrust::raw_pointer_cast(canary_c.data()), CANARY_L, wgmma_gemm_t::accumulator_t::single);
        cudaDeviceSynchronize();
        cudaMemcpy(canary_reference.data(), canary + 2 * canary_n, canary_n * sizeof(half), cudaMemcpyDeviceToHost);
    }
    // runs the canary once; false (and a message) when it fails
    const auto canary_ok = [&](const std::string &after) {
        if (canary == nullptr) {
            return true;
        }
        cudaMemset(canary + 2 * canary_n, 0xFF, canary_n * sizeof(half));
        canary_gemm.gemm(canary_plan, thrust::raw_pointer_cast(canary_c.data()), CANARY_L, wgmma_gemm_t::accumulator_t::single);
        const auto error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "canary after " << after << ": CUDA error " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        std::vector<uint16_t> result(canary_n);
        cudaMemcpy(result.data(), canary + 2 * canary_n, canary_n * sizeof(half), cudaMemcpyDeviceToHost);
        size_t differ = 0;
        for (size_t i = 0; i < canary_n; ++i) {
            differ += result[i] != canary_reference[i] ? 1 : 0;
        }
        std::cerr << "canary after " << after << ": " << (differ == 0 ? "ok" : "DIFFERS in " + std::to_string(differ))
                  << std::endl;
        return differ == 0;
    };
    canary_ok("start");
#endif
    for (auto R = std::max(1, batched_from_r); R <= GEMM::max_recursions; ++R) {
        int count = 1; // 7^R products
        for (auto r = 0; r < R; ++r) {
            count *= 7;
        }
        const auto tag = " R=" + std::to_string(R);
        for (auto L = 256; (int64_t(L) << R) < int64_t(GEMM::max_dimension); L *= 2) {
            if (!batched_only.empty()) {
                const auto key = std::to_string(R) + ":" + std::to_string(L);
                const auto list = "," + batched_only + ",";
                if (list.find("," + key + ",") == std::string::npos) {
                    continue; // not in --batched-only
                }
            }
            const auto per = size_t(L) * size_t(L); // halfs per product
            const auto bytes = 3 * size_t(count) * per * sizeof(half);
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            // a fixed margin for the cuBLAS workspace, plans and pointer arrays (was 10 % of the free
            // memory, which skipped 49 x 16384: 73.5 GiB of 78.6 GiB free)
            constexpr size_t MARGIN = size_t(2) << 30;
            if (bytes + MARGIN > free_bytes) {
                std::cerr << "batched_costs: L=" << L << " x " << count << " needs " << (bytes >> 20) << " MB, skipped"
                          << std::endl;
                continue;
            }
            half *base = nullptr;
            if (cudaMalloc(&base, bytes) != cudaSuccess) {
                cudaGetLastError();
                std::cerr << "batched_costs: cudaMalloc of " << (bytes >> 20) << " MB failed, skipped" << std::endl;
                continue;
            }
            half *A = base;
            half *B = base + size_t(count) * per;
            half *C = base + 2 * size_t(count) * per;
            cudaMemset(base, 0x3C, 2 * size_t(count) * per * sizeof(half)); // A and B
            cudaMemset(C, 0, size_t(count) * per * sizeof(half));
            // pointer arrays (host, and copies on the device) for the pointer-array calls
            const auto products = size_t(count);
            thrust::host_vector<half *> HA(products);
            thrust::host_vector<half *> HB(products);
            thrust::host_vector<half *> HC(products);
            for (size_t i = 0; i < size_t(count); ++i) {
                HA[i] = A + i * per;
                HB[i] = B + i * per;
                HC[i] = C + i * per;
            }
            const thrust::device_vector<half *> DA = HA;
            const thrust::device_vector<half *> DB = HB;
            thrust::device_vector<half *> DC = HC;
            cudaDeviceSynchronize();
            const auto stride = int64_t(per);
            if (batched_cublas) {
                cuperf_t perf("cublas_strided_batched");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &) {
                        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, L, L, L, scale.one(), A, CUDA_R_16F, L, stride,
                                                   B, CUDA_R_16F, L, stride, scale.zero(), C, CUDA_R_16F, L, stride, count,
                                                   compute_type(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    },
                    stream, measure);
                series["CUBLAS batched" + tag][L] = perf.get_avg();
            }
            if (batched_cublas) {
                cuperf_t perf("cublas_pointer_batched");
                // device arrays of the operand addresses, as cublasGemmBatchedEx takes them
                const auto *pa = reinterpret_cast<const void *const *>(thrust::raw_pointer_cast(DA.data()));
                const auto *pb = reinterpret_cast<const void *const *>(thrust::raw_pointer_cast(DB.data()));
                const auto *pc = reinterpret_cast<void *const *>(thrust::raw_pointer_cast(DC.data()));
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &) {
                        cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, L, L, L, scale.one(), pa, CUDA_R_16F, L, pb,
                                            CUDA_R_16F, L, scale.zero(), pc, CUDA_R_16F, L, count, compute_type(),
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    },
                    stream, measure);
                series["CUBLAS pointers" + tag][L] = perf.get_avg();
            }
            // cuBLASLt's fastest candidate, once for the strided batch and once with the same products
            // given as pointer arrays (CUBLASLT_BATCH_MODE_POINTER_ARRAY, as CAPS addresses its leaves)
            const auto time_best = [&](const std::string &name, const void *a, const void *b, void *c,
                                       cublasLtBatchMode_t mode) {
                try {
                    cublas_best_t best(a, b, c, uint64_t(L), int64_t(L), count, stride, s, mode);
                    if (best.ok()) {
                        cuperf_t perf(name);
                        GEMM::run_test(perf, [&](cv::cuda::Stream &) { best.run(); }, stream, measure);
                        series[name + tag][L] = perf.get_avg();
                    }
                } catch (const std::exception &e) {
                    std::cerr << name << ", L=" << L << " x " << count << ": skipped (" << e.what() << ")" << std::endl;
                }
            };
            if (batched_cublas) {
                time_best("CUBLAS best batched", A, B, C, CUBLASLT_BATCH_MODE_STRIDED);
                time_best("CUBLAS best pointers", thrust::raw_pointer_cast(DA.data()), thrust::raw_pointer_cast(DB.data()),
                          thrust::raw_pointer_cast(DC.data()), CUBLASLT_BATCH_MODE_POINTER_ARRAY);
            }
#if (LEAF_GEMM_WGMMA)
            try {
                if (batched_own && wgmma_gemm_t::supports(L)) {
                    const auto plan = wgmma_gemm_t::make_plan(HA.data(), L, HB.data(), L, L, count, nullptr);
                    wgmma_gemm_t own;
                    own.set_stream(stream);
                    own.set_staged_store(!batched_direct);
                    cuperf_t perf("own_batched");
                    GEMM::run_test(
                        perf,
                        [&](cv::cuda::Stream &) {
                            own.gemm(plan, thrust::raw_pointer_cast(DC.data()), L, GEMM::accumulator);
                        },
                        stream, measure);
                    series["OWN wgmma" + tag][L] = perf.get_avg();
                    std::cerr << "own batched, L=" << L << " x " << count << ": " << own.last_launch() << std::endl;
                    if (batched_keep_plans) {
                        kept_plans.push_back(plan);
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "own batched, L=" << L << " x " << count << ": skipped (" << e.what() << ")" << std::endl;
            }
#endif
            cudaDeviceSynchronize();
            cudaFree(base);
#if (LEAF_GEMM_WGMMA)
            if (!canary_ok("L=" + std::to_string(L) + " x " + std::to_string(count))) {
                write_series("batched_costs", "Batched GEMM Costs (7^R leaves)", GEMM::units_text(GEMM::measure), series);
                cublasDestroy_v2(handle);
                return; // the first size after which the canary fails: stop here
            }
#endif
        }
        // written after every R, so that a crash in a later batch keeps the batches measured so far
        write_series("batched_costs", "Batched GEMM Costs (7^R leaves)", GEMM::units_text(GEMM::measure), series);
    }
    cublasDestroy_v2(handle);
}

static void malloc_time()
{
    const auto test_name = "Initialization time";
    std::map<int, std::map<int, stat_t>> statistcs;
    cv::cuda::Stream stream;
    for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
        if (!GEMM::inputs_fit(N)) {
            break;
        }
        cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1);
        cv::cuda::GpuMat B_dev(N, N / 2, CV_32FC1);
        for (auto R = 1; R <= GEMM::max_recursions; ++R) {
            if (!GEMM::is_valid(N, R)) {
                continue;
            }
            perf_t perf("init");
            for (auto i = 0; i < GEMM::iterations; ++i) {
                if (i >= GEMM::warm_ups) {
                    perf.start();
                }
                BFSCaps bfs;
                bfs.initialize(A_dev, B_dev, R, stream);
                if (i >= GEMM::warm_ups) {
                    perf.end();
                }
            }

            stat_t stat;
            stat.maxVal = perf.get_max();
            stat.minVal = perf.get_min();
            stat.avgVal  = perf.get_avg();
            statistcs[R][N] = stat;
        }
    }

    std::map<test_type_t, stats_map_t> stats;
    stats[PERFORMACE][1] = statistcs; // one "range": R -> N -> statistics
    write_stats("initialization_time", test_name, GEMM::UNITS_S<GEMM::TIME_M>::text(), stats); // CPU timer
}

static void comminication_costs()
{
    cv::cuda::Stream stream;
    std::map<int, std::map<int, double>> measured_cost;
    {
        for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
            if (!GEMM::inputs_fit(N)) {
                break;
            }
            cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1), B_dev(N, N / 2, CV_32FC1);
            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                BFSCaps bfs;
                if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                    continue;
                }

                cuperf_t perf("communication");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &stream) {
                        bfs.forward_pass(stream);
                        bfs.backward_pass(stream);
                    },
                    stream, measure);
                measured_cost[R][N] = perf.get_avg();
            }
        }
    }

    std::map<std::string, std::map<int, double>> series;
    for (const auto &it : measured_cost) {
        series["R=" + std::to_string(it.first)] = it.second;
    }
    write_series("communication_costs", "Communication Costs", GEMM::units_text(GEMM::measure), series);
}

static void warp_and_L2_usage()
{
    cv::cuda::Stream stream;
    std::map<int, std::map<int, double>> measured_cost;
    {
        for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
            if (!GEMM::inputs_fit(N)) {
                break;
            }
            cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1), B_dev(N, N / 2, CV_32FC1);
            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                BFSCaps bfs;
                if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                    continue;
                }

                cuperf_t perf("warp_and_L2");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &stream) {
                        bfs.batched_leaf(stream, GEMM::accumulator);
                    },
                    stream, measure);
                measured_cost[R][N] = perf.get_avg();
            }
        }
    }

    std::map<std::string, std::map<int, double>> series;
    for (const auto &it : measured_cost) {
        series["R=" + std::to_string(it.first)] = it.second;
    }
    write_series("warp_and_l2_usage", "Warp and L2 Usage", GEMM::units_text(GEMM::measure), series);
}

static void batched_gemm_vs_gemm()
{
    cv::cuda::Stream stream;
    std::map<int, double> target_cost;
    std::map<int, std::map<int, double>> measured_cost;
    {
        cublas_gemm_t cgemm(stream);
        for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
            if (!GEMM::inputs_fit(N)) {
                break;
            }
            cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1), B_dev(N, N / 2, CV_32FC1),
                C_dev(N, N / 2, CV_32FC1);
            {
                cuperf_t perf("single_gemm");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &stream) { default_gemm(cgemm, stream, A_dev, B_dev, C_dev); },
                    stream, measure);
                target_cost[N] = perf.get_avg();
            }
            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                BFSCaps bfs;
                if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                    continue;
                }

                cuperf_t perf("batched_gemm");
                GEMM::run_test(
                    perf,
                    [&](cv::cuda::Stream &stream) {
                        bfs.batched_leaf(stream, GEMM::accumulator);
                    },
                    stream, measure);
                measured_cost[R][N] = perf.get_avg();
            }
        }
    }

    std::map<std::string, std::map<int, double>> series;
    series["Single GEMM"] = target_cost;
    for (const auto &it : measured_cost) {
        series["Batched GEMM R=" + std::to_string(it.first)] = it.second;
    }
    write_series("batched_gemm_vs_gemm", "Batched GEMM vs Single GEMM", GEMM::units_text(GEMM::measure), series);
}

static bool precision = false;       ///< --precision: measure the rounding errors (precision_errors())
static int precision_max_dim = 16384; ///< --precision-max-dim: largest N of precision_errors()

/// N x N half matrix (CV_16F) <-> the N x N / 2 CV_32FC1 GpuMat the GEMMs take (pitch kept)
static void upload_half(const cv::Mat &H, cv::cuda::GpuMat &D, cv::cuda::Stream &stream)
{
    const cv::Mat view(H.rows, H.cols / 2, CV_32FC1, const_cast<uchar *>(H.data), H.step);
    D.upload(view, stream);
}

static cv::Mat download_half(const cv::cuda::GpuMat &D, cv::cuda::Stream &stream)
{
    cv::Mat view;
    D.download(view, stream);
    stream.waitForCompletion();
    return cv::Mat(view.rows, view.cols * 2, CV_16F, view.data, view.step).clone();
}

/**
 * @brief Rounding errors of cuBLAS, our GEMM, CAPS and DFSCaps, f16 and f32 accumulation
 * @details Inputs uniform in [-1, 1], rounded to half; the reference is the double product of the
 * rounded inputs (cublasDgemm on the GPU). Per result: the norm-wise relative error
 * ||C^ - C||_F / ||C||_F and the largest element error relative to max |C|. Writes
 * <results_dir>/precision.csv. A and B are uploaded again before every run (the encoder of CAPS
 * overwrites them; the upload keeps the buffers, so the BFS address tables stay valid).
 */
static void precision_errors()
{
    cv::cuda::Stream stream;
    cublas_gemm_t cgemm(stream);
    std::filesystem::create_directories(results_dir);
    const auto path = std::filesystem::path(results_dir) / "precision.csv";
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    out << "N,R,schedule,accumulator,rel_fro,rel_max" << std::endl;
    const wgmma_gemm_t::accumulator_t accumulators[] = {wgmma_gemm_t::accumulator_t::half,
                                                      wgmma_gemm_t::accumulator_t::single};
    cv::theRNG().state = 1;
    for (auto N = 1024; N <= precision_max_dim && N < GEMM::max_dimension; N *= 2) {
        cv::Mat A16;
        cv::Mat B16;
        {
            cv::Mat X(N, N, CV_32F);
            cv::randu(X, -1.0, 1.0);
            X.convertTo(A16, CV_16F);
            cv::randu(X, -1.0, 1.0);
            X.convertTo(B16, CV_16F);
        }
        cv::Mat ref; // N x N double
        {
            cv::Mat A64;
            cv::Mat B64;
            A16.convertTo(A64, CV_64F);
            B16.convertTo(B64, CV_64F);
            cv::cuda::GpuMat dA(A64);
            cv::cuda::GpuMat dB(B64);
            cv::cuda::GpuMat dC(N, N, CV_64F);
            cgemm.set_stream(stream);
            const double one = 1.0;
            const double zero = 0.0;
            const auto ld = int(dA.step / sizeof(double));
            const auto ldc = int(dC.step / sizeof(double));
            // row-major C = A B is the column-major C^T = B^T A^T
            if (cublasDgemm_v2(cgemm.handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
                               reinterpret_cast<const double *>(dB.data), ld, reinterpret_cast<const double *>(dA.data),
                               ld, &zero, reinterpret_cast<double *>(dC.data), ldc) != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "precision: cublasDgemm failed, N=" << N << std::endl;
                break;
            }
            dC.download(ref, stream);
            stream.waitForCompletion();
        }
        const auto ref_fro = cv::norm(ref, cv::NORM_L2);
        const auto ref_max = cv::norm(ref, cv::NORM_INF);
        const auto report = [&](int R, const std::string &schedule, const cv::Mat &C16) {
            cv::Mat C;
            C16.convertTo(C, CV_64F);
            const auto fro = cv::norm(C, ref, cv::NORM_L2) / ref_fro;
            const auto max = cv::norm(C, ref, cv::NORM_INF) / ref_max;
            const auto acc = f32_accumulation() ? "f32" : "f16";
            out << N << "," << R << "," << schedule << "," << acc << "," << fro << "," << max << std::endl;
            std::cerr << "precision N=" << N << " R=" << R << " " << schedule << " " << acc << ": rel_fro " << fro
                      << ", rel_max " << max << std::endl;
        };
        cv::cuda::GpuMat A_dev(N, N / 2, CV_32FC1);
        cv::cuda::GpuMat B_dev(N, N / 2, CV_32FC1);
        cv::cuda::GpuMat C_dev(N, N / 2, CV_32FC1);
        for (const auto acc : accumulators) {
            GEMM::accumulator = acc;
            upload_half(A16, A_dev, stream);
            upload_half(B16, B_dev, stream);
            default_gemm(cgemm, stream, A_dev, B_dev, C_dev);
            report(0, "cublas", download_half(C_dev, stream));
#if (LEAF_GEMM_WGMMA)
            if (wgmma_gemm_t::supports(N)) {
                own_gemm_t(A_dev, B_dev, C_dev, stream).run(acc);
                report(0, "own", download_half(C_dev, stream));
            }
#endif
            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
#if (LEAF_GEMM_WGMMA)
                if (dfs_caps && R >= 2 && (N >> R) >= 256) try { // before is_valid(): it checks the BFS memory
                    DFSCaps dfs;
                    if (dfs.initialize(A_dev, B_dev, R, stream)) {
                        upload_half(A16, A_dev, stream);
                        upload_half(B16, B_dev, stream);
                        dfs.run(stream, acc);
                        report(R, "dfs_caps", download_half(dfs.get_result(), stream));
                    }
                } catch (const std::exception &e) {
                    std::cerr << "skip: DFSCaps precision N=" << N << ", R=" << R << ": " << e.what() << std::endl;
                }
#endif
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                {
                    BFSCaps bfs;
                    if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                        continue;
                    }
                    upload_half(A16, A_dev, stream);
                    upload_half(B16, B_dev, stream);
                    bfs.forward_pass(stream);
                    bfs.batched_leaf(stream, acc);
                    bfs.backward_pass(stream);
                    report(R, "caps", download_half(bfs.get_result(), stream));
                } // one BFS instance at a time (__constant__ address tables)
            }
        }
    }
    GEMM::accumulator = wgmma_gemm_t::accumulator_t::half;
    std::cerr << "precision: " << path.string() << std::endl;
}

} // namespace GEMM

/**
 * @brief Main entry point for GEMM benchmark application
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @return Exit status code
 * 
 * Parses the command line and runs the benchmark suite; results go to CSV files
 * (results.hpp), python py/plots.py draws them.
 * @see GEMM::help() for command line usage information
 */
int main(int argc, char *argv[])
{
    print_build_config(std::cerr);
    GEMM::device_name = report_device_capabilities();
    QCoreApplication app(argc, argv);
    GEMM::options_t options("GEMM benchmark: performance and precision of cuBLAS and CAPS + cuBLAS; "
                            "results as CSV files for python py/plots.py");
    options.add("max-dim", "largest N (exclusive)", GEMM::max_dimension)
        .add("max-range", "largest value range of the random data", GEMM::max_range)
        .add("max-recursions", "largest R", GEMM::max_recursions)
        .add("int-range", "1: integer random data", 0)
        .add("regenerate", "1: regenerate the stored test data", 0)
        .add("warm-ups", "untimed runs before measuring", GEMM::warm_ups)
        .add("warm-up-ms", "then keep running for at least this many ms before measuring (power cap "
                           "steady state; 0: off)", 1000)
        .add("iterations", "timed runs", GEMM::iterations)
        .add("keep", "1: keep generated test data", 1)
        .add("results", "folder of the result CSV files", GEMM::results_dir)
        .add("measure", "time (ms per call) or energy (mJ per call, NVML counter: N >= 8192 only)",
             std::string("time"))
        .add("sizes", "comma separated N for the timings (any k * 2^R, e.g. 49152,98304); empty: powers of two", std::string(""))
        .add("dfs-caps", "1: also time DFSCaps (depth-first top level, products added into C; R >= 2)", 1)
        .add("batched", "1: also time the batched leaf GEMMs alone (batched_costs.csv: ours vs cuBLAS batched)", 1)
        .add("relative-costs", "1: time cuBLAS / ours / CAPS per N (relative_costs.csv); 0: skip it", 1)
        .add("batched-from-r", "first R (batch 7^R) of the batched benchmark", 1)
        .add("batched-own", "1: time our wgmma in the batched benchmark; 0: cuBLAS only", 1)
        .add("batched-keep-plans", "1: keep our plans (tensor maps) alive until the batched benchmark ends", 0)
        .add("batched-cublas", "1: time cuBLAS's batched GEMMs in the batched benchmark; 0: ours only", 1)
        .add("batched-direct", "1: our wgmma with the direct store of C in the batched benchmark", 0)
        .add("batched-canary", "1: check a fixed small batch of ours after every size (finds a size that corrupts memory)", 0)
        .add("batched-only", "only these batched sizes, R:L,R:L,... (empty: all)", std::string(""))
        .add("accumulator", "f16 or f32: accumulation of every timed GEMM (cuBLAS, ours, CAPS leaves)", std::string("f16"))
        .add("precision", "1: measure the rounding errors (precision.csv; f16 and f32) and skip the timings", 0)
        .add("precision-max-dim", "largest N of --precision", 16384);
    options.parse(app);

    GEMM::max_dimension = options.get_int("max-dim");
    GEMM::max_range = options.get_int("max-range");
    GEMM::max_recursions = options.get_int("max-recursions");
    GEMM::int_range = options.get_int("int-range") != 0;
    GEMM::regenerate = options.get_int("regenerate") != 0;
    GEMM::warm_ups = options.get_int("warm-ups");
    GEMM::warm_up_ms = options.get_int("warm-up-ms");
    GEMM::iterations = options.get_int("iterations");
    GEMM::keep = options.get_int("keep");
    GEMM::results_dir = options.get_string("results");
    {
        std::stringstream list(options.get_string("sizes"));
        for (std::string item; std::getline(list, item, ',');) {
            if (!item.empty()) {
                GEMM::sizes.push_back(std::stoi(item));
            }
        }
    }
    GEMM::dfs_caps = options.get_int("dfs-caps") != 0;
    GEMM::batched = options.get_int("batched") != 0;
    GEMM::relative = options.get_int("relative-costs") != 0;
    GEMM::batched_from_r = options.get_int("batched-from-r");
    GEMM::batched_own = options.get_int("batched-own") != 0;
    GEMM::batched_keep_plans = options.get_int("batched-keep-plans") != 0;
    GEMM::batched_cublas = options.get_int("batched-cublas") != 0;
    GEMM::batched_direct = options.get_int("batched-direct") != 0;
    GEMM::batched_canary = options.get_int("batched-canary") != 0;
    GEMM::batched_only = options.get_string("batched-only");
    {
        const auto accumulator = options.get_string("accumulator");
        if (accumulator != "f16" && accumulator != "f32") {
            std::cerr << "--accumulator must be f16 or f32, got " << accumulator << std::endl;
            return 1;
        }
        GEMM::accumulator = accumulator == "f32" ? wgmma_gemm_t::accumulator_t::single : wgmma_gemm_t::accumulator_t::half;
    }
    GEMM::precision = options.get_int("precision") != 0;
    GEMM::precision_max_dim = options.get_int("precision-max-dim");
    {
        const auto measure = options.get_string("measure");
        if (measure != "time" && measure != "energy") {
            std::cerr << "--measure must be time or energy, got " << measure << std::endl;
            return 1;
        }
        GEMM::measure = measure == "time" ? GEMM::TIME_M : GEMM::ENERGY_M;
    }

    GEMM::help(argv[0]);
    
    if (GEMM::precision) {
        GEMM::precision_errors();
        return 0;
    }
    if (1) {
        if (GEMM::relative) {
            GEMM::relative_costs();
        }
        if (GEMM::batched) {
            GEMM::batched_costs();
        }
        return 0;
    }

    if (0) {
        GEMM::warp_and_L2_usage();
    }
    if (1) {
        GEMM::batched_gemm_vs_gemm();
    }
    if (1) {
        GEMM::comminication_costs();
    }

    if (0) {
        GEMM::malloc_time();
    }

    // Run performance tests
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> cublas_vs_mkl;
    GEMM::run_cublas_performance_tests(cublas_vs_mkl);

    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> caps_cublas_vs_mkl;
    GEMM::run_caps_cublas_performance_tests(caps_cublas_vs_mkl);

    // plots: python py/plots.py (reads both files)
    GEMM::write_stats("cublas", "CUBLAS", GEMM::units_text(GEMM::measure), cublas_vs_mkl);
    GEMM::write_stats("caps_cublas", "CAPS+CUBLAS", GEMM::units_text(GEMM::measure), caps_cublas_vs_mkl);

    cv::HMat::Release();
    return 0;
}
