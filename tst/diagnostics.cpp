/**
 * @file diagnostics.cpp
 * @brief Launch-configuration diagnostics (not a test; not registered with CTest)
 *
 * Moved out of unit_tests.cpp, where these used to be toggled with if (0) blocks.
 *     ./bin/test_diagnostics bandwidth [max_GiB [iterations]]
 *     ./bin/test_diagnostics bulk-reduce [iterations]
 *         TMA bulk store vs bulk f16 reduce-add of 16 KB tiles into 8 MiB .. 1 GiB (fused-reconstruct feasibility)
 *     ./bin/test_diagnostics l2-reuse [max_MiB [iterations]]
 *         L2 capacity for reuse with the passes' cache operators: read after read / after write, 4 .. max MiB
 *         measured DRAM bandwidth (device-to-device copy and memset, 256 MiB .. max) vs theoretical
 *     ./bin/test_diagnostics caps-compare N R loopsA loopsB [rounds [iterations]]
 *         full CAPS with two loop settings "e,b,r,d[,grid_divisor]", both accumulators, A/B alternating
 *     ./bin/test_diagnostics tune [warm_ups iterations [table [from_N [from_R [to_R]]]]]
 *         table: default <source dir>/gen/tuning_table.hpp (GEMM_SOURCE_DIR, set by tst/CMakeLists.txt)
 *         tunes this GPU and writes its entries into gen/tuning_table.hpp (include/tuning.hpp): BFS
 *         loop counts in a -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF build, the mma leaf GEMM always.
 *         The table is saved after every (N, R); from_N / from_R resume an interrupted run there.
 *         to_R: the largest R tuned at every N (default 4), e.g. 16384 2 2 tunes N=16384, R=2 only.
 *     ./bin/test_diagnostics gemm-compare N [iterations [g]]
 *         cuBLAS (half) and our wgmma GEMM (half and float accumulation; clusters "c1" / "c2";
 *         both schedules: "1/blk" one tile per block, "pers" persistent) on one N x N product
 *         (R = 0), same operands; ms,
 *         TFLOPS and wgmma vs cuBLAS (cuBLAS timed again right before every wgmma line, against
 *         the clock drift of a warming GPU; "direct": set_staged_store(false);
 *         "t256" / "t128": set_tile_n(), the others choose
 *         the tile width automatically, see "tile" in the launch line). g: wgmma
 *         tile-order group size (0, the default: automatic;
 *         -1: sweep 4 .. 256). Small enough to run under a profiler:
 *             ncu --kernel-name regex:"gemm|wgmma" ./bin/test_diagnostics gemm-compare 4096 3
 *     ./bin/test_diagnostics batched-compare L count [iterations]
 *         A batch of count products of L x L (the leaves of CAPS: 49 x 1024, 343 x 2048, ...):
 *         cuBLAS strided batched (compute 16F) against our batched wgmma in several schedules
 *         (one tile per block, persistent with several g, clusters, 128-column tiles), half
 *         accumulation, cuBLAS timed again before every line (power cap), after ~1 s of warm-up.
 *         For the automatic schedule's batch rule (it looks at the tiles per product only).
 * The GPU-free analyses (l2-warps, select-nodes) are Python: python py/diagnostics.py
 */

#include <cmath>
#include <cstdio>
#include <opencv2/core.hpp>

#include <bfs_caps.hpp>
#include <dfs_caps.hpp>
#include <common.hpp>
#include <cublas_gemm.hpp>
#include <cublasLt.h>
#include <wgmma_gemm.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <tuning.hpp>
#include <wgmma_gemm.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <iostream>

#ifdef GEMM_SOURCE_DIR // tst/CMakeLists.txt
static const std::string DEFAULT_TABLE = GEMM_SOURCE_DIR "/gen/tuning_table.hpp";
#else
static const std::string DEFAULT_TABLE = "../gen/tuning_table.hpp";
#endif


/// ms per call of f after warm_ups calls: the median of TIMING_BATCHES batches of
/// iterations / TIMING_BATCHES calls (one slow batch - a clock change, another process - does
/// not decide). Throws on a CUDA error, so that a failed kernel never looks like a fast one.
static constexpr int TIMING_BATCHES = 5;
static double time_ms(cv::cuda::Stream &stream, int warm_ups, int iterations, const std::function<void()> &f)
{
    for (auto i = 0; i < warm_ups; ++i) {
        f();
    }
    const auto s = (cudaStream_t) stream.cudaPtr();
    const auto per_batch = std::max(1, iterations / TIMING_BATCHES);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> times;
    auto error = cudaSuccess;
    for (auto b = 0; b < TIMING_BATCHES && cudaSuccess == error; ++b) {
        cudaEventRecord(start, s);
        for (auto i = 0; i < per_batch; ++i) {
            f();
        }
        cudaEventRecord(stop, s);
        error = cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms / per_batch);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (cudaSuccess != error) {
        throw std::runtime_error(std::string("CUDA error while timing: ") + cudaGetErrorString(error));
    }
    std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
    return times[times.size() / 2];
}

/// A new choice must be faster by more than this to replace the current one (this GPU's table
/// entry or the default tile order): timing noise alone does not change the table
static constexpr double TIE = 0.01;

static std::string loops_str(const BFSCaps::config_t &c)
{
    return std::to_string(c.m_transform_A_and_B_loops_count) + "," + std::to_string(c.m_bilinear_loops_count) + ","
           + std::to_string(c.m_reconstruct_loops_count) + "," + std::to_string(c.m_transform_C_loops_count);
}

// Tunes every supported (N, R) on this GPU and writes its entries into the table file
// (gen/tuning_table.hpp, include/tuning.hpp):
//   - BFS loop counts (only in a build with -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF): the whole CAPS
//     pipeline (forward, batched cuBLAS leaf GEMM, backward) is timed with the current loop counts
//     (this GPU's table entry, else "default", else the heuristic) and with the tuned ones, and
//     the faster set is written (the tuned one only when it is more than TIE faster)
// Rebuild afterwards (with LOOPS_COUNT_AS_TEMPLATE_ARGS=ON for production) to compile it in.
static void tune_all(int warm_ups, int iterations, const std::string &table_path, int from_N, int from_R,
                     int to_R)
{
    const auto &gpu = tuning::device_name();
    tuning::bfs_entries_t bfs_entries;
    tuning::bfs_notes_t bfs_notes;   // measured times, written as comments after the entries
    // "+12.3%": b relative to a
    const auto percent = [](double b, double a) {
        std::ostringstream ss;
        ss << std::showpos << std::fixed << std::setprecision(1) << 100.0 * (b - a) / a << "%";
        return ss.str();
    };
    const auto tune_loops = !LOOPS_COUNT_AS_TEMPLATE_ARGS;
    if (!tune_loops) {
        std::cerr << "LOOPS_COUNT_AS_TEMPLATE_ARGS=ON build: BFS loop counts are compiled in and not tuned "
                     "(build with OFF for that)"
                  << std::endl;
        return;
    }
    std::cout << "tuning for \"" << gpu << "\"" << std::endl;

    // the table is rewritten after every (N, R), so that an interrupted run (a crash, a power
    // loss) keeps what it measured; if table_path cannot be written, ./tuning_table.hpp is used
    std::string written_path = table_path;
    const auto save = [&]() {
        for (;;) {
            try {
                tuning::write(written_path, gpu, bfs_entries, bfs_notes);
                return true;
            } catch (const std::exception &e) {
                const std::string fallback = "tuning_table.hpp";
                std::cerr << e.what() << std::endl;
                if (written_path == fallback) {
                    return false;
                }
                std::cerr << "writing this GPU's entries to ./" << fallback << " instead" << std::endl;
                written_path = fallback;
            }
        }
    };

    for (auto N = 256; N <= 131072; N *= 2) { // the N of the select_kernel() table; what does not fit is skipped
        for (auto R = 1; R <= std::min(4, to_R); ++R) {
            const auto order = N >> R;
            if (order <= 256) {
                continue;
            }
            if (N < from_N || (N == from_N && R < from_R)) {
                continue; // resumed run
            }
            std::cout << "N=" << N << " R=" << R << ": tuning" << std::endl;
            // skip what this GPU cannot hold (A, B and the BFS buffers) instead of failing in the middle
            {
                std::string why;
                if (!BFSCaps::fits_device_memory(N, R, &why)) {
                    std::cerr << "skip: " << why << std::endl;
                    continue;
                }
            }
            cv::cuda::Stream stream;
            cv::cuda::GpuMat A_dev, B_dev; // initialize() needs continuous N x N/2 buffers
            BFSCaps bfs;
            try {
                // not cv::cuda::createContinuous(): its int element count overflows at N = 65536
                // (2^31 floats), see create_continuous() in common.hpp
                create_continuous(N, N / 2, CV_32FC1, A_dev);
                create_continuous(N, N / 2, CV_32FC1, B_dev);
                A_dev.setTo(cv::Scalar::all(0), stream); // defined contents (half zeros)
                B_dev.setTo(cv::Scalar::all(0), stream);
                if (!bfs.initialize(A_dev, B_dev, R, stream)) {
                    std::cerr << "skip N=" << N << ", R=" << R << ": initialize failed" << std::endl;
                    continue;
                }
            } catch (const cv::Exception &e) {
                std::cerr << "skip N=" << N << ", R=" << R << ": " << e.err << std::endl;
                continue;
            } catch (const std::exception &e) { // e.g. thrust::system::bad_alloc
                std::cerr << "skip N=" << N << ", R=" << R << ": " << e.what() << std::endl;
                continue;
            }
            stream.waitForCompletion();
            const auto caps = [&] {
                bfs.forward_pass(stream);
                bfs.batched_leaf(stream, wgmma_gemm_t::accumulator_t::half);
                bfs.backward_pass(stream);
            };

            try {
                if (tune_loops) {
                    const auto current = bfs.get_config();
                    const auto before = time_ms(stream, warm_ups, iterations, caps);
                    const auto tuned = bfs.tune(warm_ups, iterations, true); // applies
                    const auto after = time_ms(stream, warm_ups, iterations, caps);
                    const auto &keep = after < before * (1.0 - TIE) ? tuned : current;
                    std::cout << "N=" << N << " R=" << R << ": CAPS " << before << " ms with loops "
                              << loops_str(current) << ", " << after << " ms tuned " << loops_str(tuned) << " -> keep "
                              << loops_str(keep) << std::endl;
                    bfs_entries[{N, R}] = {int(keep.m_transform_A_and_B_loops_count),
                                           int(keep.m_bilinear_loops_count),
                                           int(keep.m_reconstruct_loops_count),
                                           int(keep.m_transform_C_loops_count)};
                    // grid divisor sweep with the kept loop counts: cooperative grid = co-resident
                    // blocks / divisor. Not stored in the table yet: printed (and noted) so that a
                    // fixed value or a table field can be decided from the numbers.
                    std::ostringstream divisors;
                    {
                        bfs.set_config(keep);
                        auto best_divisor = 0u;
                        auto best_divisor_ms = 0.0;
                        auto default_ms = 0.0;
                        std::cout << "N=" << N << " R=" << R << ": grid divisor, forward / backward / CAPS ms:";
                        for (const auto divisor : {1u, 2u, 3u, 4u, 6u, 8u}) {
                            bfs.set_grid_divisor(divisor);
                            const auto fwd = time_ms(stream, warm_ups, iterations, [&] { bfs.forward_pass(stream); });
                            const auto bwd = time_ms(stream, warm_ups, iterations, [&] { bfs.backward_pass(stream); });
                            const auto total = time_ms(stream, warm_ups, iterations, caps);
                            std::cout << std::setprecision(4) << "  " << divisor << ": " << fwd << " / " << bwd << " / "
                                      << total;
                            if (divisor == BFSCaps::config_t::DEFAULT_GRID_DIVISOR) {
                                default_ms = total;
                            }
                            if (best_divisor == 0 || total < best_divisor_ms) {
                                best_divisor = divisor;
                                best_divisor_ms = total;
                            }
                        }
                        bfs.set_grid_divisor(BFSCaps::config_t::DEFAULT_GRID_DIVISOR);
                        std::cout << " -> best " << best_divisor << std::endl;
                        divisors << std::setprecision(4) << "; grid divisor " << best_divisor << ": " << best_divisor_ms
                                 << " ms (" << BFSCaps::config_t::DEFAULT_GRID_DIVISOR << ": " << default_ms << " ms)";
                    }
                    {
                        std::ostringstream note;
                        note << std::setprecision(4) << "CAPS " << before << " ms with " << loops_str(current)
                             << ", " << after << " ms tuned " << loops_str(tuned) << divisors.str();
                        bfs_notes[{N, R}] = note.str();
                    }
                }

            } catch (const std::exception &e) { // cv::Exception, thrust::system::bad_alloc, ...
                std::cerr << "skip N=" << N << ", R=" << R << ": " << e.what() << std::endl;
            }
            if (!bfs_entries.empty()) {
                save();
            }
        }
    }
    if (bfs_entries.empty()) {
        std::cerr << "nothing tuned, " << table_path << " unchanged" << std::endl;
        return;
    }
    if (written_path != table_path) {
        std::cout << "./" << written_path << ": " << bfs_entries.size() << " BFS entries for \"" << gpu << "\"; copy them into gen/tuning_table.hpp" << std::endl;
        return;
    }
    std::cout << table_path << ": " << bfs_entries.size() << " BFS entries for \"" << gpu << "\"; rebuild to compile them in" << std::endl;
}

/**
 * @brief gemm-compare: cuBLAS vs our wgmma GEMM on one N x N half product (see the file comment)
 * @details The operands are filled with the byte 0x3C (every half 0x3C3C, about 1.06): defined,
 * finite and the same for both. Timing as in tune (time_ms(): median of TIMING_BATCHES batches
 * after 2 warm-up calls). wgmma is skipped where it cannot run (no CC 9.0 / sm_90a code, N not a
 * multiple of 256).
 */
static void gemm_compare(int N, int iterations, int g)
{
    cv::cuda::Stream stream;
    cv::cuda::GpuMat A, B, C;
    create_continuous(N, N / 2, CV_32FC1, A); // N x N halves; size_t sizes up to N = 65536
    create_continuous(N, N / 2, CV_32FC1, B);
    create_continuous(N, N / 2, CV_32FC1, C);
    cudaMemset(A.data, 0x3C, A.step * A.rows);
    cudaMemset(B.data, 0x3C, B.step * B.rows);
    cudaMemset(C.data, 0, C.step * C.rows);
    cudaDeviceSynchronize();

    const auto flops = 2.0 * double(N) * double(N) * double(N);
    const auto report = [&](const std::string &what, double ms, double reference) {
        std::cout << "N=" << N << "  " << std::left << std::setw(30) << what << std::right << std::fixed
                  << std::setprecision(4) << std::setw(11) << ms << " ms " << std::setprecision(1) << std::setw(7)
                  << flops / (ms * 1e9) << " TFLOPS";
        if (reference > 0) {
            std::cout << "  " << std::showpos << 100.0 * (ms - reference) / reference << std::noshowpos
                      << "% vs cuBLAS " << std::setprecision(4) << reference << " ms";
        }
        std::cout << std::defaultfloat << std::endl;
    };

    cublas_gemm_t cgemm(stream); // cublasHgemm (half compute), as in the costs benchmark
    const auto cublas = time_ms(stream, 2, iterations, [&] { cgemm.gemm(A, B, C); });
    report("cuBLAS half", cublas, -1);

    std::string why;
    if (!wgmma_gemm_t::available(why) || !wgmma_gemm_t::supports(N)) {
        std::cout << "N=" << N << "  wgmma: n/a (" << (why.empty() ? "N not a multiple of 256" : why) << ")" << std::endl;
        return;
    }
    // one-entry pointer arrays, filled on the host (see gemm_benchmark.cpp), and the tensor maps
    const thrust::host_vector<half *> HA(1, reinterpret_cast<half *>(A.data));
    const thrust::host_vector<half *> HB(1, reinterpret_cast<half *>(B.data));
    const thrust::host_vector<half *> HC(1, reinterpret_cast<half *>(C.data));
    const thrust::device_vector<half *> DA = HA;
    const thrust::device_vector<half *> DB = HB;
    thrust::device_vector<half *> DC = HC;
    const auto ld = int(A.step / sizeof(half));
    const auto ldc = int(C.step / sizeof(half));
    const auto plan = wgmma_gemm_t::make_plan(DA, ld, DB, ld, N);
    wgmma_gemm_t wgmma;
    wgmma.set_stream(stream);
    {
        // host time per call (enqueue only, no synchronization inside): for small N the GPU time is
        // a few us, so a call that takes longer on the host bounds the rate of back-to-back calls
        constexpr int HOST_CALLS = 200;
        const auto host_us = [&](const std::function<void()> &f) {
            f();
            stream.waitForCompletion();
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < HOST_CALLS; ++i) {
                f();
            }
            const auto end = std::chrono::steady_clock::now();
            stream.waitForCompletion();
            return std::chrono::duration<double, std::micro>(end - begin).count() / HOST_CALLS;
        };
        const auto cublas_us = host_us([&] { cgemm.gemm(A, B, C); });
        const auto wgmma_us = host_us([&] { wgmma.gemm(plan, DA, DC, ldc, wgmma_gemm_t::accumulator_t::half); });
        std::cout << "N=" << N << "  host per call (enqueue): cuBLAS " << std::fixed << std::setprecision(2) << cublas_us
                  << " us, wgmma " << wgmma_us << " us" << std::defaultfloat << std::endl;
    }
    const auto groups = g >= 0 ? std::vector<int>{g} : std::vector<int>{4, 8, 16, 32, 64, 128, 256};
    using schedule_t = wgmma_gemm_t::schedule_t;
    struct config_t
    {
        int cluster;
        schedule_t schedule;
        bool round_sync;
        bool staged; // set_staged_store(): staged (the default) or direct store of C
        int tile_n;  // set_tile_n(): automatic (the default: 128 for small batches) or forced 256 / 128
        const char *label;
    };
    constexpr int tile_auto = wgmma_gemm_t::AUTO_TILE_N;
    const config_t configs[] = {
        {1, schedule_t::one_tile_per_block, false, true, tile_auto, "1/blk "},
        {1, schedule_t::one_tile_per_block, false, true, 256, "1/blk t256 "},
        {1, schedule_t::one_tile_per_block, false, true, 128, "1/blk t128 "},
        {1, schedule_t::one_tile_per_block, false, false, tile_auto, "1/blk direct "},
        {1, schedule_t::persistent, false, true, tile_auto, "pers "},
        {1, schedule_t::persistent, false, true, 128, "pers t128 "},
        {1, schedule_t::persistent, false, false, tile_auto, "pers direct "},
        {1, schedule_t::persistent, true, true, tile_auto, "pers sync "}, // round barrier (set_round_sync())
        {2, schedule_t::one_tile_per_block, false, true, tile_auto, "1/blk "},
        {2, schedule_t::persistent, false, true, tile_auto, "pers "},
        {2, schedule_t::persistent, false, false, tile_auto, "pers direct "},
    };
    for (const auto &config : configs) {
        wgmma.set_cluster(config.cluster);
        wgmma.set_schedule(config.schedule);
        wgmma.set_round_sync(config.round_sync);
        wgmma.set_staged_store(config.staged);
        wgmma.set_tile_n(config.tile_n);
        // the plain variants without the automatic barrier, so that "pers" vs "pers sync" compares
        wgmma.set_round_sync_threshold(0);
        wgmma.set_round_sync_work(0);
        for (const auto group : groups) {
            wgmma.set_order(group);
            for (const auto accumulator : {wgmma_gemm_t::accumulator_t::half, wgmma_gemm_t::accumulator_t::single}) {
                // cuBLAS again right before every wgmma timing (interleaved): long sweeps heat the
                // GPU and its clock drops, so a single cuBLAS time from the start (cool GPU) made
                // the later configs look slower (H100, N = 8192: the first g of every config was
                // the fastest by 3 - 6 %); each line compares with the cuBLAS time it was taken with
                const auto reference = time_ms(stream, 2, iterations, [&] { cgemm.gemm(A, B, C); });
                const auto t = time_ms(stream, 2, iterations, [&] { wgmma.gemm(plan, DA, DC, ldc, accumulator); });
                const auto name = "c" + std::to_string(config.cluster) + " " + config.label
                                  + (accumulator == wgmma_gemm_t::accumulator_t::half ? "half " : "float")
                                  + " g=" + (group > 0 ? std::to_string(group) : std::string("auto"));
                report(name, t, reference);
            }
            std::cout << "        launch: " << wgmma.last_launch() << std::endl;
        }
    }
    wgmma.set_round_sync(wgmma_gemm_t::DEFAULT_ROUND_SYNC);
    wgmma.set_round_sync_threshold(wgmma_gemm_t::DEFAULT_ROUND_SYNC_THRESHOLD);
    wgmma.set_round_sync_work(wgmma_gemm_t::DEFAULT_ROUND_SYNC_WORK);
    wgmma.set_staged_store(wgmma_gemm_t::DEFAULT_STAGED_STORE);
    wgmma.set_tile_n(wgmma_gemm_t::DEFAULT_TILE_N);
}

/**
 * @brief cublas-algos: every candidate of cuBLASLt's heuristic for one N x N half GEMM, timed
 * @details Column-major N x N operands (square, so the layout does not change the work), the same
 * fill as gemm-compare, 64 MB workspace (cuBLAS's default upper end). Each candidate is tried once
 * before timing; one that fails (status != success) is reported and skipped. Rank 0 is the
 * heuristic's first choice, i.e. what cublasHgemm / cublasGemmEx use.
 */
/**
 * @brief batched-stress (see the file comment): does our batched GEMM write outside C or go wrong
 *        when it is launched many times in a row?
 */
/**
 * @brief batched-compare (see the file comment): which schedule of ours suits a batch of leaves?
 */
static void batched_compare(int L, int count, int iterations)
{
    std::string why;
    if (!wgmma_gemm_t::available(why) || !wgmma_gemm_t::supports(L)) {
        std::cout << "batched-compare: wgmma n/a (" << (why.empty() ? "L not a multiple of 256" : why) << ")" << std::endl;
        return;
    }
    const auto per = size_t(L) * size_t(L);
    const auto n = size_t(count) * per;
    half *base = nullptr;
    if (cudaMalloc(&base, 3 * n * sizeof(half)) != cudaSuccess) {
        cudaGetLastError();
        std::cout << "batched-compare: cudaMalloc of " << (3 * n * sizeof(half) >> 20) << " MB failed" << std::endl;
        return;
    }
    half *A = base;
    half *B = A + n;
    half *C = B + n;
    cudaMemset(base, 0x3C, 2 * n * sizeof(half)); // every half ~1.06: finite
    cudaMemset(C, 0, n * sizeof(half));
    const auto products = size_t(count); // named: HA(size_t(count)) would declare a function
    thrust::host_vector<half *> HA(products);
    thrust::host_vector<half *> HB(products);
    thrust::host_vector<half *> HC(products);
    for (size_t i = 0; i < products; ++i) {
        HA[i] = A + i * per;
        HB[i] = B + i * per;
        HC[i] = C + i * per;
    }
    const thrust::device_vector<half *> DA = HA;
    const thrust::device_vector<half *> DB = HB;
    thrust::device_vector<half *> DC = HC;
    const auto plan = wgmma_gemm_t::make_plan(DA, L, DB, L, L);
    cv::cuda::Stream stream;
    cublasHandle_t handle = nullptr;
    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, (cudaStream_t) stream.cudaPtr());
    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f);
    const auto stride = int64_t(per);
    const auto cublas = [&] {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, L, L, L, &one, A, CUDA_R_16F, L, stride, B,
                                   CUDA_R_16F, L, stride, &zero, C, CUDA_R_16F, L, stride, count, CUBLAS_COMPUTE_16F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    // ~1 s of warm-up: the power cap's steady clock for everything below
    {
        const auto begin = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - begin < std::chrono::milliseconds(1000)) {
            cublas();
            stream.waitForCompletion();
        }
    }
    const auto flops = 2.0 * double(L) * double(L) * double(L) * double(count);
    const auto first = time_ms(stream, 2, iterations, cublas);
    std::cout << "L=" << L << " x " << count << "  cuBLAS batched " << std::fixed << std::setprecision(4) << first
              << " ms " << std::setprecision(1) << flops / (first * 1e9) << " TFLOPS" << std::defaultfloat << std::endl;
    wgmma_gemm_t wgmma;
    wgmma.set_stream(stream);
    using schedule_t = wgmma_gemm_t::schedule_t;
    struct config_t
    {
        int cluster;
        schedule_t schedule;
        int order;   // set_order(): 0 automatic
        int tile_n;  // set_tile_n(): automatic, 256 or 128
        bool sync;   // set_round_sync()
        const char *label;
    };
    constexpr int tile_auto = wgmma_gemm_t::AUTO_TILE_N;
    const config_t configs[] = {
        {1, schedule_t::automatic, 0, tile_auto, false, "c1 automatic"},
        {1, schedule_t::one_tile_per_block, 0, tile_auto, false, "c1 1/blk g=auto"},
        {1, schedule_t::persistent, 0, tile_auto, false, "c1 pers g=auto"},
        {1, schedule_t::persistent, 4, tile_auto, false, "c1 pers g=4"},
        {1, schedule_t::persistent, 8, tile_auto, false, "c1 pers g=8"},
        {1, schedule_t::persistent, 16, tile_auto, false, "c1 pers g=16"},
        {1, schedule_t::persistent, 0, tile_auto, true, "c1 pers sync g=auto"},
        {1, schedule_t::one_tile_per_block, 0, 128, false, "c1 1/blk t128"},
        {1, schedule_t::persistent, 0, 128, false, "c1 pers t128"},
        {2, schedule_t::one_tile_per_block, 0, tile_auto, false, "c2 1/blk g=auto"},
        {2, schedule_t::persistent, 0, tile_auto, false, "c2 pers g=auto"},
    };
    for (const auto &config : configs) {
        wgmma.set_cluster(config.cluster);
        wgmma.set_schedule(config.schedule);
        wgmma.set_order(config.order);
        wgmma.set_tile_n(config.tile_n);
        wgmma.set_round_sync(config.sync);
        wgmma.set_round_sync_threshold(0); // only where asked ("sync")
        wgmma.set_round_sync_work(0);
        const auto reference = time_ms(stream, 2, iterations, cublas);
        const auto t = time_ms(stream, 2, iterations,
                               [&] { wgmma.gemm(plan, DA, DC, L, wgmma_gemm_t::accumulator_t::half); });
        std::cout << "L=" << L << " x " << count << "  " << std::left << std::setw(22) << config.label << std::right
                  << std::fixed << std::setprecision(4) << std::setw(10) << t << " ms " << std::setprecision(1)
                  << std::setw(6) << flops / (t * 1e9) << " TFLOPS  " << std::showpos << 100.0 * (t - reference) / reference
                  << std::noshowpos << "% vs cuBLAS " << std::setprecision(4) << reference << " ms" << std::defaultfloat
                  << "   " << wgmma.last_launch() << std::endl;
    }
    // both accumulators of ours (default configuration)
    using accumulator_t = wgmma_gemm_t::accumulator_t;
    const auto print = [&](const char *label, accumulator_t accumulator, double t, double reference) {
        std::cout << "L=" << L << " x " << count << "  " << std::left << std::setw(26) << label
                  << (accumulator == accumulator_t::half ? " f16" : " f32") << std::right << std::fixed
                  << std::setprecision(4) << std::setw(10) << t << " ms " << std::setprecision(1) << std::setw(6)
                  << flops / (t * 1e9) << " TFLOPS  " << std::showpos << 100.0 * (t - reference) / reference
                  << std::noshowpos << "% vs cuBLAS " << std::setprecision(4) << reference << " ms" << std::defaultfloat;
    };
    wgmma.set_cluster(1);
    wgmma.set_schedule(schedule_t::automatic);
    wgmma.set_order(0);
    wgmma.set_tile_n(tile_auto);
    wgmma.set_round_sync(false);
    for (const auto accumulator : {accumulator_t::half, accumulator_t::single}) {
        const auto reference = time_ms(stream, 2, iterations, cublas);
        const auto t = time_ms(stream, 2, iterations, [&] { wgmma.gemm(plan, DA, DC, L, accumulator); });
        print("ours (automatic)", accumulator, t, reference);
        std::cout << std::endl;
    }
    cublasDestroy_v2(handle);
    cudaFree(base);
}

// caps-compare N R loopsA loopsB [rounds [iterations]]
// Full CAPS (forward pass, leaf GEMM, backward pass) with two loop-count settings
// "encode,bilinear,reconstruct,decode[,grid_divisor]", for both accumulators (f16, f32). The
// settings alternate A B B A ... per round, so slow drifts (clocks, temperature) hit both alike.
// Prints every round, then mean / standard deviation / min per accumulator and setting and B vs A.
// bfs.tune() times every pass alone (no leaf GEMM), so it cannot see an accumulator dependence;
// this mode measures it end to end. Needs LOOPS_COUNT_AS_TEMPLATE_ARGS=OFF (run-time loop counts).
static bool parse_loops(const char *text, int N, int R, BFSCaps::config_t &c)
{
    unsigned int v[5] = {0, 0, 0, 0, 0};
    const int n = std::sscanf(text, "%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4]);
    if (n < 4 || 0 == v[0] || 0 == v[1] || 0 == v[2] || 0 == v[3]) {
        return false;
    }
    c.m_transform_A_and_B_loops_count = v[0];
    c.m_bilinear_loops_count = v[1];
    c.m_reconstruct_loops_count = v[2];
    c.m_transform_C_loops_count = v[3];
    c.m_bilinear_rows_per_loop = N / v[1];
    c.m_reconstruct_rows_per_loop = (N >> R) / v[2];
    if (5 == n && v[4] > 0) {
        c.m_grid_divisor = v[4];
    }
    return true;
}

static void caps_compare(int N, int R, const char *loops_a, const char *loops_b, int rounds, int iterations)
{
#if (LOOPS_COUNT_AS_TEMPLATE_ARGS)
    (void) N, (void) R, (void) loops_a, (void) loops_b, (void) rounds, (void) iterations;
    std::cerr << "caps-compare needs a build with LOOPS_COUNT_AS_TEMPLATE_ARGS=OFF" << std::endl;
#else
    cv::cuda::Stream stream;
    cv::cuda::GpuMat A_dev, B_dev; // initialize() needs continuous N x N/2 buffers
    create_continuous(N, N / 2, CV_32FC1, A_dev);
    create_continuous(N, N / 2, CV_32FC1, B_dev);
    A_dev.setTo(cv::Scalar::all(0), stream);
    B_dev.setTo(cv::Scalar::all(0), stream);
    BFSCaps bfs;
    if (!bfs.initialize(A_dev, B_dev, R, stream)) {
        std::cerr << "caps-compare: initialize failed, N=" << N << ", R=" << R << std::endl;
        return;
    }
    stream.waitForCompletion();
    BFSCaps::config_t config[2] = {bfs.get_config(), bfs.get_config()};
    if (!parse_loops(loops_a, N, R, config[0]) || !parse_loops(loops_b, N, R, config[1])) {
        std::cerr << "caps-compare: loops as encode,bilinear,reconstruct,decode[,grid_divisor], got \"" << loops_a
                  << "\" \"" << loops_b << "\"" << std::endl;
        return;
    }
    using acc_t = wgmma_gemm_t::accumulator_t;
    const acc_t accumulators[2] = {acc_t::half, acc_t::single};
    const char *acc_names[2] = {"f16", "f32"};
    std::vector<double> times[2][2]; // [accumulator][setting]
    const auto name = [&](int s) {
        return std::string(s ? "B " : "A ") + loops_str(config[s]) + " g" + std::to_string(config[s].m_grid_divisor);
    };
    for (int round = 0; round < rounds; ++round) {
        for (int a = 0; a < 2; ++a) {
            for (int k = 0; k < 2; ++k) {
                const int s = (round & 1) ? 1 - k : k; // A B, then B A
                bfs.set_config(config[s]);
                const auto ms = time_ms(stream, 1, iterations, [&] {
                    bfs.forward_pass(stream);
                    bfs.batched_leaf(stream, accumulators[a]);
                    bfs.backward_pass(stream);
                });
                times[a][s].push_back(ms);
                std::cout << "round " << round << " " << acc_names[a] << " " << name(s) << ": " << ms << " ms"
                          << std::endl;
            }
        }
    }
    double mean[2][2];
    for (int a = 0; a < 2; ++a) {
        for (int s = 0; s < 2; ++s) {
            const auto &t = times[a][s];
            double sum = 0, sum2 = 0, low = t.empty() ? 0 : t[0];
            for (const auto x : t) {
                sum += x;
                sum2 += x * x;
                low = std::min(low, x);
            }
            const auto n = double(t.size());
            mean[a][s] = sum / n;
            const auto sd = n > 1 ? std::sqrt(std::max(0.0, (sum2 - sum * sum / n) / (n - 1))) : 0.0;
            std::cout << "summary N=" << N << " R=" << R << " " << acc_names[a] << " " << name(s) << ": mean "
                      << mean[a][s] << " ms, sd " << sd << " ms, min " << low << " ms (" << t.size() << " rounds)"
                      << std::endl;
        }
        std::cout << "summary N=" << N << " R=" << R << " " << acc_names[a] << ": B vs A "
                  << 100.0 * (mean[a][1] / mean[a][0] - 1.0) << " %" << std::endl;
    }
#endif
}

// bandwidth [max_GiB [iterations]]
// The DRAM bandwidth this GPU really delivers, the ceiling for the memory-bound BFS passes:
// device-to-device copies (every byte read and written once, like the passes; counted as 2 x
// bytes) and memsets (write only) of 256 MiB .. max_GiB, all far beyond the L2, against the
// theoretical 2 x memory clock x bus width. CUDA runtime calls only (cudaMemcpyAsync and
// cudaMemsetAsync are themselves well-tuned kernels).
static void bandwidth(int max_gib, int iterations)
{
    int device = 0, clock_khz = 0, bus_bits = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    const double peak_gbs = 2.0 * clock_khz * 1e3 * (bus_bits / 8) / 1e9;
    std::cout << "bandwidth: theoretical " << peak_gbs << " GB/s (memory clock " << clock_khz / 1000
              << " MHz, bus " << bus_bits << " bit)" << std::endl;
    cv::cuda::Stream stream;
    const auto s = (cudaStream_t) stream.cudaPtr();
    for (size_t mib = 256; mib <= size_t(max_gib) * 1024; mib *= 2) {
        const size_t bytes = mib << 20;
        void *a = nullptr, *b = nullptr;
        if (cudaSuccess != cudaMalloc(&a, bytes) || cudaSuccess != cudaMalloc(&b, bytes)) {
            cudaGetLastError(); // clear
            cudaFree(a);
            std::cout << "bandwidth: " << mib << " MiB x 2 do not fit" << std::endl;
            break;
        }
        cudaMemsetAsync(a, 1, bytes, s);
        cudaMemsetAsync(b, 2, bytes, s);
        const auto copy_ms = time_ms(stream, 2, iterations, [&] { cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, s); });
        const auto set_ms = time_ms(stream, 2, iterations, [&] { cudaMemsetAsync(b, 3, bytes, s); });
        const auto copy_gbs = 2.0 * bytes / (copy_ms * 1e-3) / 1e9;
        const auto set_gbs = double(bytes) / (set_ms * 1e-3) / 1e9;
        std::cout << std::fixed << std::setprecision(1) << "bandwidth " << std::setw(6) << mib << " MiB:  copy "
                  << std::setw(7) << copy_gbs << " GB/s (" << std::setw(4) << 100.0 * copy_gbs / peak_gbs
                  << " % of theoretical), memset " << std::setw(7) << set_gbs << " GB/s (" << std::setw(4)
                  << 100.0 * set_gbs / peak_gbs << " %)" << std::defaultfloat << std::endl;
        cudaFree(a);
        cudaFree(b);
    }
}

// l2-reuse [max_MiB [iterations]]
// How much data the L2 really keeps for reuse, with the cache operators of the passes
// (__stcg / __ldcg, 16 B per thread; common.hpp l2_probe()), for X = 4, 8, ... max_MiB:
//   read after read:   the same X read again and again (hits while X stays in L2)
//   read after write:  X written, then read back (one level of a pass feeding the next)
// Both as the read's GB/s; a 1 GiB read of another buffer before every write evicts X. The
// GB/s fall from L2 to DRAM speed where X stops fitting: the reuse capacity, to compare with
// the 50 MiB LRU cache of the L2 simulator (py/gemm_l2).
static void l2_reuse(int max_mib, int iterations)
{
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cv::cuda::Stream stream;
    const auto s = (cudaStream_t) stream.cudaPtr();
    const size_t flush_bytes = size_t(1) << 30;
    void *buffer = nullptr, *flush = nullptr;
    int *sink = nullptr;
    if (cudaSuccess != cudaMalloc(&buffer, size_t(max_mib) << 20) || cudaSuccess != cudaMalloc(&flush, flush_bytes)
        || cudaSuccess != cudaMalloc(&sink, 8 * sms * sizeof(int))) {
        std::cout << "l2-reuse: cudaMalloc failed" << std::endl;
        return;
    }
    l2_probe(true, flush, flush_bytes, nullptr, sms, s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const auto timed = [&](const std::function<void()> &f) { // ms of f, launched alone
        cudaEventRecord(e0, s);
        f();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return double(ms);
    };
    std::cout << "l2-reuse: " << sms << " SMs; GB/s of the read (median of " << iterations << ")" << std::endl;
    for (int mib = 4; mib <= max_mib; mib += mib < 32 ? 4 : 8) {
        const size_t bytes = size_t(mib) << 20;
        std::vector<double> rar, raw;
        l2_probe(true, buffer, bytes, nullptr, sms, s);
        l2_probe(false, buffer, bytes, sink, sms, s); // warm
        for (int i = 0; i < iterations; ++i) {
            rar.push_back(timed([&] { l2_probe(false, buffer, bytes, sink, sms, s); }));
        }
        for (int i = 0; i < iterations; ++i) {
            l2_probe(false, flush, flush_bytes, sink, sms, s); // evict X
            l2_probe(true, buffer, bytes, nullptr, sms, s);
            raw.push_back(timed([&] { l2_probe(false, buffer, bytes, sink, sms, s); }));
        }
        std::sort(rar.begin(), rar.end());
        std::sort(raw.begin(), raw.end());
        const auto gbs = [&](double ms) { return double(bytes) / (ms * 1e-3) / 1e9; };
        std::cout << std::fixed << std::setprecision(1) << "l2-reuse " << std::setw(4) << mib
                  << " MiB:  read after read " << std::setw(7) << gbs(rar[rar.size() / 2]) << " GB/s,  read after write "
                  << std::setw(7) << gbs(raw[raw.size() / 2]) << " GB/s" << std::defaultfloat << std::endl;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(buffer);
    cudaFree(flush);
    cudaFree(sink);
}

// bulk-reduce [iterations]
// The epilogue operation of a fused leaf reconstruction: a 16 KB tile written with the TMA bulk
// copy (store) against the bulk reduce-add (f16 add into what is there) to a target of X bytes,
// X from 8 MiB (in L2) to 1 GiB (DRAM). The target is reset (memset 0) before every timed pass;
// GB/s count the tile bytes written (a reduce-add from DRAM also reads them).
static void bulk_reduce(int iterations)
{
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cv::cuda::Stream stream;
    const auto s = (cudaStream_t) stream.cudaPtr();
    const size_t max_bytes = size_t(1) << 30;
    void *dst = nullptr;
    if (cudaSuccess != cudaMalloc(&dst, max_bytes)) {
        std::cout << "bulk-reduce: cudaMalloc failed" << std::endl;
        return;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::cout << "bulk-reduce: " << sms << " SMs, 4 blocks per SM, 16 KB tiles, median of " << iterations << std::endl;
    for (const size_t mib : {8, 16, 24, 32, 48, 64, 128, 1024}) {
        const size_t bytes = mib << 20;
        double ms[2];
        for (int reduce = 0; reduce < 2; ++reduce) {
            std::vector<double> t;
            for (int i = 0; i <= iterations; ++i) {
                cudaMemsetAsync(dst, 0, bytes, s);
                cudaEventRecord(e0, s);
                bulk_probe(reduce != 0, dst, bytes, sms, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float m = 0;
                cudaEventElapsedTime(&m, e0, e1);
                if (i > 0) { // the first one is a warm-up
                    t.push_back(m);
                }
            }
            std::sort(t.begin(), t.end());
            ms[reduce] = t[t.size() / 2];
        }
        const auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "bulk-reduce: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        const auto gbs = [&](double m) { return double(bytes) / (m * 1e-3) / 1e9; };
        std::cout << std::fixed << std::setprecision(1) << "bulk-reduce " << std::setw(5) << mib << " MiB:  store "
                  << std::setw(7) << gbs(ms[0]) << " GB/s (" << std::setprecision(3) << ms[0] << " ms),  reduce-add "
                  << std::setprecision(1) << std::setw(7) << gbs(ms[1]) << " GB/s (" << std::setprecision(3) << ms[1]
                  << " ms),  reduce / store " << std::setprecision(2) << ms[1] / ms[0] << std::defaultfloat << std::endl;
    }
    // one value check: after one reduce pass on zeros every half is 2^-10
    cudaMemsetAsync(dst, 0, 16384, s);
    bulk_probe(true, dst, 16384, 1, s);
    std::vector<unsigned short> host(8192);
    cudaMemcpyAsync(host.data(), dst, 16384, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    int wrong = 0;
    for (const auto v : host) {
        wrong += v != 0x1400; // 2^-10 as half
    }
    std::cout << "bulk-reduce check: " << (wrong ? std::to_string(wrong) + " wrong halfs" : std::string("ok")) << std::endl;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dst);
}

/**
 * @brief reduce-check: the leaf GEMM with reduce-add epilogue (wgmma_gemm_t::gemm_reduce()) against
 * products stored and added on the host, for the two-level decoding W x W of the alternative basis
 * @details 7 A and 7 B blocks of L x L; product p = 7 i + j is A_i * B_j and adds into the blocks of the
 * 4L x 4L result C that column i of W (outer) and column j (inner) name, with the product of their
 * signs: C11 = m1 + m2, C12 = m5 - m7, C21 = m3 + m6, C22 = m5 + m6 - m2 - m4 (src/reconstruct.cuh).
 * Checks 64 rows of C, then times: storing the 49 products (today's leaf GEMM), reduce-add of all 49
 * in one launch, and 7 launches of 7 (the depth-first top level).
 */
static void reduce_check(int L, int iterations, bool f32)
{
    std::string why;
    if (!wgmma_gemm_t::available(why) || !wgmma_gemm_t::supports(L)) {
        std::cout << "reduce-check: wgmma n/a (" << (why.empty() ? "L not a multiple of 256" : why) << ")" << std::endl;
        return;
    }
    using acc_t = wgmma_gemm_t::accumulator_t;
    const auto accumulator = f32 ? acc_t::single : acc_t::half;
    const auto per = size_t(L) * size_t(L);
    const auto ld = 4 * L; // C is 4L x 4L
    const size_t halves = 14 * per + 16 * per + 49 * per;
    half *base = nullptr;
    if (cudaMalloc(&base, halves * sizeof(half)) != cudaSuccess) {
        cudaGetLastError();
        std::cout << "reduce-check: cudaMalloc of " << (halves * sizeof(half) >> 20) << " MB failed" << std::endl;
        return;
    }
    half *A = base, *B = A + 7 * per, *C = B + 7 * per, *P = C + 16 * per;
    {
        // one random block (+ 7 x 1031 extra halves), each operand uploaded from its own offset
        std::vector<half> host(per + 7 * 1031 + 7 * 977);
        uint32_t x = 12345;
        for (auto &h : host) {
            x = x * 1664525u + 1013904223u;
            h = __float2half(float(int(x >> 9) - (1 << 22)) / float(1 << 22)); // uniform in [-1, 1)
        }
        for (int i = 0; i < 7; ++i) {
            cudaMemcpy(A + i * per, host.data() + i * 1031, per * sizeof(half), cudaMemcpyHostToDevice);
            cudaMemcpy(B + i * per, host.data() + 7 * 1031 + i * 977, per * sizeof(half), cudaMemcpyHostToDevice);
        }
    }
    // W: per product m1 .. m7, its (quadrant row, column, sign) list
    struct term_t { int r, c, s; };
    const std::vector<std::vector<term_t>> W = {{{0, 0, 1}},
                                                {{0, 0, 1}, {1, 1, -1}},
                                                {{1, 0, 1}},
                                                {{1, 1, -1}},
                                                {{0, 1, 1}, {1, 1, 1}},
                                                {{1, 0, 1}, {1, 1, 1}},
                                                {{0, 1, -1}}};
    constexpr int T = wgmma_gemm_t::MAX_REDUCE_TARGETS;
    thrust::host_vector<half *> HA(49), HB(49), HP(49), HT(49 * T, nullptr);
    struct target_t { int br, bc, s; };
    std::vector<std::vector<target_t>> targets(49);
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            const int p = 7 * i + j;
            HA[p] = A + i * per;
            HB[p] = B + j * per;
            HP[p] = P + p * per;
            int t = 0;
            for (const auto &o : W[i]) {
                for (const auto &u : W[j]) {
                    const target_t g{2 * o.r + u.r, 2 * o.c + u.c, o.s * u.s};
                    targets[p].push_back(g);
                    auto ptr = reinterpret_cast<uintptr_t>(C + size_t(g.br) * L * ld + size_t(g.bc) * L);
                    HT[p * T + t++] = reinterpret_cast<half *>(ptr | (g.s < 0 ? 1 : 0));
                }
            }
        }
    }
    const thrust::device_vector<half *> DA = HA, DB = HB;
    thrust::device_vector<half *> DP = HP;
    const thrust::device_vector<half *> DT = HT;
    const auto plan = wgmma_gemm_t::make_plan(DA, L, DB, L, L);
    // one launch per outer product i: plans over 7 products each, targets from row 7 i T
    std::vector<wgmma_gemm_t::plan_t> plans7;
    for (int i = 0; i < 7; ++i) {
        plans7.push_back(wgmma_gemm_t::make_plan(HA.data() + 7 * i, L, HB.data() + 7 * i, L, L, 7, &HA[7 * i]));
    }
    cv::cuda::Stream stream;
    const auto s = (cudaStream_t) stream.cudaPtr();
    wgmma_gemm_t wgmma;
    wgmma.set_stream(stream);
    const auto Tp = thrust::raw_pointer_cast(DT.data());
    const auto store = [&] { wgmma.gemm(plan, thrust::raw_pointer_cast(DP.data()), L, accumulator); };
    const auto reduce = [&] {
        cudaMemsetAsync(C, 0, 16 * per * sizeof(half), s);
        wgmma.gemm_reduce(plan, Tp, T, ld, accumulator);
    };
    const auto reduce7 = [&] {
        cudaMemsetAsync(C, 0, 16 * per * sizeof(half), s);
        for (int i = 0; i < 7; ++i) {
            wgmma.gemm_reduce(plans7[i], Tp + 7 * i * T, T, ld, accumulator);
        }
    };
    // check
    store();
    reduce();
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        std::cout << "reduce-check: CUDA error " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaFree(base);
        return;
    }
    std::cout << "reduce-check L=" << L << (f32 ? " f32" : " f16") << " acc: " << wgmma.last_launch() << std::endl;
    double worst = 0;
    int bad = 0;
    uint32_t x = 777;
    std::vector<half> row(ld), prow(L);
    for (int k = 0; k < 64; ++k) {
        x = x * 1664525u + 1013904223u;
        const int R = int(x % uint32_t(ld));
        const int br = R / L, rr = R % L;
        cudaMemcpy(row.data(), C + size_t(R) * ld, ld * sizeof(half), cudaMemcpyDeviceToHost);
        std::vector<double> sum(ld, 0.0), mag(ld, 0.0);
        for (int p = 0; p < 49; ++p) {
            bool loaded = false;
            for (const auto &g : targets[p]) {
                if (g.br != br) {
                    continue;
                }
                if (!loaded) {
                    cudaMemcpy(prow.data(), P + p * per + size_t(rr) * L, L * sizeof(half), cudaMemcpyDeviceToHost);
                    loaded = true;
                }
                for (int c = 0; c < L; ++c) {
                    const double v = __half2float(prow[c]);
                    sum[g.bc * L + c] += g.s * v;
                    mag[g.bc * L + c] += std::fabs(v);
                }
            }
        }
        for (int c = 0; c < ld; ++c) {
            const double e = std::fabs(__half2float(row[c]) - sum[c]) / (mag[c] + 1e-3);
            worst = std::max(worst, e);
            bad += e > 4e-3;
        }
    }
    std::cout << "  64 rows: worst error / sum |terms| " << std::scientific << std::setprecision(2) << worst
              << std::defaultfloat << ", " << bad << " entries above 4e-3 -> " << (bad == 0 ? "OK" : "FAILED") << std::endl;
    const auto flops = 2.0 * double(L) * double(L) * double(L) * 49;
    const auto report = [&](const char *name, const std::function<void()> &f) {
        const auto ms = time_ms(stream, 2, iterations, f);
        std::cout << "  " << std::left << std::setw(28) << name << std::right << std::fixed << std::setprecision(3) << ms
                  << " ms " << std::setprecision(1) << flops / (ms * 1e9) << " TFLOPS" << std::defaultfloat << std::endl;
    };
    report("store 49 products", store);
    report("memset C only", [&] { cudaMemsetAsync(C, 0, 16 * per * sizeof(half), s); });
    report("reduce 49, il1 (+memset)", reduce);
    report("reduce 7 x 7 (+memset)", reduce7);
    wgmma.set_round_sync(true); // a launch of 7 has too little work per CTA for the automatic barrier
    report("reduce 7 x 7 sync (+memset)", reduce7);
    wgmma.set_round_sync(false);
    cudaFree(base);
}

/// Sampled error of C against the reference: max |C - ref| / max |ref| over 64 rows (both N x N halves)
static double sampled_error(const cv::cuda::GpuMat &C, const cv::cuda::GpuMat &ref, int N)
{
    std::vector<half> a(N), b(N);
    double worst = 0, scale = 0;
    uint32_t x = 4242;
    for (int k = 0; k < 64; ++k) {
        x = x * 1664525u + 1013904223u;
        const int r = int(x % uint32_t(N));
        cudaMemcpy(a.data(), C.ptr<float>(r), N * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), ref.ptr<float>(r), N * sizeof(half), cudaMemcpyDeviceToHost);
        for (int c = 0; c < N; ++c) {
            const double u = __half2float(a[c]), v = __half2float(b[c]);
            worst = std::max(worst, std::fabs(u - v));
            scale = std::max(scale, std::fabs(v));
        }
    }
    return worst / std::max(scale, 1e-30);
}

/**
 * @brief dfs-check: DFSCaps (depth-first top level, leaf products reduce-added into C) against cuBLAS,
 * and BFS CAPS of the same R when it fits: sampled error, memory and time
 */
static void dfs_check(int N, int R, int iterations, bool f32)
{
    using acc_t = wgmma_gemm_t::accumulator_t;
    const auto accumulator = f32 ? acc_t::single : acc_t::half;
    cv::cuda::Stream stream;
    std::cout << "dfs-check N=" << N << " R=" << R << (f32 ? " f32" : " f16") << " acc; memory: DFSCaps serial "
              << (DFSCaps::required_memory(N, R, DFSCaps::mode_t::serial) >> 20) << " MB, overlapped "
              << (DFSCaps::required_memory(N, R, DFSCaps::mode_t::overlapped) >> 20) << " MB, BFS "
              << (BFSCaps::required_memory(N, R) >> 20) << " MB, plain GEMM " << ((3 * size_t(N) * N * sizeof(half)) >> 20)
              << " MB" << std::endl;
    cv::cuda::GpuMat ref;
    create_continuous(N, N / 2, CV_32FC1, ref);
    double ref_ms = 0;
    {
        std::srand(1);
        auto data = generate_half_data(N, 1, false, true, false, stream);
        stream.waitForCompletion();
        cublas_gemm_t cublas(stream);
        cublas.gemm(data.first, data.second, ref);
        stream.waitForCompletion();
        cv::cuda::GpuMat tmp;
        create_continuous(N, N / 2, CV_32FC1, tmp);
        ref_ms = time_ms(stream, 1, iterations, [&] { cublas.gemm(data.first, data.second, tmp); });
    }
    std::cout << "  cuBLAS (cublasHgemm)        " << std::fixed << std::setprecision(3) << ref_ms << " ms" << std::defaultfloat << std::endl;
    struct dfs_config_t
    {
        const char *name;
        DFSCaps::mode_t mode;
        bool priority;
        int form_sms;
    };
    const dfs_config_t configs[] = {
        {"serial", DFSCaps::mode_t::serial, false, 0},
        {"overlapped, priority", DFSCaps::mode_t::overlapped, true, 0},
        {"overlapped, 4 SMs free", DFSCaps::mode_t::overlapped, false, 4},
        {"overlapped, 8 SMs free", DFSCaps::mode_t::overlapped, false, 8},
        {"overlapped, 16 SMs free", DFSCaps::mode_t::overlapped, false, 16},
        {"overlapped, prio + 8 SMs", DFSCaps::mode_t::overlapped, true, 8},
    };
    for (const auto &config : configs) {
        std::srand(1);
        auto data = generate_half_data(N, 1, false, true, false, stream);
        stream.waitForCompletion();
        DFSCaps dfs;
        dfs.set_form_priority(config.priority);
        std::string why;
        if (!dfs.initialize(data.first, data.second, R, stream, &why, config.mode)) {
            std::cout << "  DFSCaps " << config.name << ": " << why << std::endl;
            continue;
        }
        dfs.set_form_sms(config.form_sms);
        dfs.run(stream, accumulator);
        if (cudaStreamSynchronize((cudaStream_t) stream.cudaPtr()) != cudaSuccess) {
            std::cout << "  DFSCaps " << config.name << ": CUDA error " << cudaGetErrorString(cudaGetLastError()) << std::endl;
            return;
        }
        size_t free_bytes = 0, total = 0;
        cudaMemGetInfo(&free_bytes, &total);
        const auto err = sampled_error(dfs.get_result(), ref, N);
        const auto ms = time_ms(stream, 1, iterations, [&] { dfs.run(stream, accumulator); });
        std::cout << "  DFSCaps R=" << R << " " << std::left << std::setw(26) << config.name << std::right << std::fixed
                  << std::setprecision(3) << ms << " ms, " << ms / ref_ms << " x cuBLAS, error " << std::scientific
                  << std::setprecision(2) << err << std::defaultfloat << ", device in use " << ((total - free_bytes) >> 20)
                  << " MB" << std::endl;
    }
    std::string why;
    if (!BFSCaps::fits_device_memory(N, R, &why)) {
        std::cout << "  BFS: " << why << std::endl;
        return;
    }
    std::srand(1);
    auto data = generate_half_data(N, 1, false, true, false, stream);
    stream.waitForCompletion();
    BFSCaps bfs;
    if (!bfs.initialize(data.first, data.second, R, stream)) {
        std::cout << "  BFS: initialize failed" << std::endl;
        return;
    }
    const auto pipeline = [&] {
        bfs.forward_pass(stream);
        bfs.batched_leaf(stream, accumulator);
        bfs.backward_pass(stream);
    };
    pipeline();
    stream.waitForCompletion();
    size_t free_bytes = 0, total = 0;
    cudaMemGetInfo(&free_bytes, &total);
    const auto err = sampled_error(bfs.get_result(), ref, N);
    const auto ms = time_ms(stream, 1, iterations, pipeline);
    std::cout << "  BFS CAPS R=" << R << "            " << std::fixed << std::setprecision(3) << ms << " ms, "
              << ms / ref_ms << " x cuBLAS, error " << std::scientific << std::setprecision(2) << err << std::defaultfloat
              << ", device in use " << ((total - free_bytes) >> 20) << " MB" << std::endl;
}

static int arg(int argc, char *argv[], int i, int fallback)
{
    return i < argc ? std::atoi(argv[i]) : fallback;
}

int main(int argc, char *argv[])
{
    print_build_config(std::cerr);
    const char *mode = argc > 1 ? argv[1] : "";
    report_device_capabilities();
    if (0 == std::strcmp(mode, "dfs-check")) {
        dfs_check(arg(argc, argv, 2, 16384), arg(argc, argv, 3, 2), arg(argc, argv, 4, 5), argc > 5 && 0 == std::strcmp(argv[5], "f32"));
    } else if (0 == std::strcmp(mode, "reduce-check")) {
        reduce_check(arg(argc, argv, 2, 8192), arg(argc, argv, 3, 10), argc > 4 && 0 == std::strcmp(argv[4], "f32"));
    } else if (0 == std::strcmp(mode, "bulk-reduce")) {
        bulk_reduce(arg(argc, argv, 2, 21));
    } else if (0 == std::strcmp(mode, "l2-reuse")) {
        l2_reuse(arg(argc, argv, 2, 96), arg(argc, argv, 3, 21));
    } else if (0 == std::strcmp(mode, "bandwidth")) {
        bandwidth(arg(argc, argv, 2, 16), arg(argc, argv, 3, 20));
    } else if (0 == std::strcmp(mode, "caps-compare")) {
        caps_compare(arg(argc, argv, 2, 65536), arg(argc, argv, 3, 2), argc > 4 ? argv[4] : "", argc > 5 ? argv[5] : "",
                     arg(argc, argv, 6, 10), arg(argc, argv, 7, 3));
    } else if (0 == std::strcmp(mode, "tune")) {
        tune_all(arg(argc, argv, 2, 5), arg(argc, argv, 3, 30), argc > 4 ? std::string(argv[4]) : DEFAULT_TABLE,
                 arg(argc, argv, 5, 0), arg(argc, argv, 6, 0), arg(argc, argv, 7, 4));
    } else if (0 == std::strcmp(mode, "gemm-compare")) {
        gemm_compare(arg(argc, argv, 2, 8192), arg(argc, argv, 3, 20), arg(argc, argv, 4, 0));
    } else if (0 == std::strcmp(mode, "batched-compare")) {
        batched_compare(arg(argc, argv, 2, 1024), arg(argc, argv, 3, 49), arg(argc, argv, 4, 20));
    } else {
        std::cerr << "usage: " << argv[0]
                  << " tune [warm_ups iterations [table [from_N [from_R [to_R]]]]]"
                     " | caps-compare N R loopsA loopsB [rounds [iterations]] | gemm-compare N [iterations [g]]"
                     " | batched-compare L count [iterations] | reduce-check [L [iterations [f16|f32]]]"
                     " | dfs-check [N [R [iterations [f16|f32]]]]"
                     " | bandwidth [max_GiB [iterations]] | l2-reuse [max_MiB [iterations]] | bulk-reduce [iterations]"
                     "  (l2-warps, select-nodes: py/diagnostics.py)"
                  << std::endl;
        return 1;
    }
    return 0;
}
