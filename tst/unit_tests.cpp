/**
 * @file unit_tests.cpp
 * @brief QtTest suite for the GPU pipeline (BFSCaps, DFSCaps, cuBLAS)
 *
 * Pure CPU checks of the reference live in reference.cpp; the diagnostic tools that
 * used to be commented in/out of main() live in diagnostics.cpp.
 *
 * GPU tests are skipped (not failed) when no CUDA device is available.
 *     ./bin/test_unit_tests                 all tests
 *     ./bin/test_unit_tests caps            one test (all data rows)
 *     ./bin/test_unit_tests caps:N=8192,R=2,range=1
 *     ./bin/test_unit_tests wgmmaMatchesCublas                   our Hopper leaf GEMM (H100 only)
 */

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <bfs_caps.hpp>
#include <common.hpp>
#include <cublas_gemm.hpp>
#include <wgmma_gemm.hpp>
#include <hmat.hpp>
#include <dfs_caps.hpp>
#include <reference.hpp>
#include "test_utils.hpp"

#include <QtTest/QtTest>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#ifndef Q_MOC_RUN // Qt's moc cannot parse CUDA 13 Thrust (CCCL 3 namespace macros)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {

bool has_cuda_device()
{
    int count = 0;
    return cudaSuccess == cudaGetDeviceCount(&count) && count > 0;
}

std::string name_of(const char *test, int N, int R, int range)
{
    std::stringstream ss;
    ss << test << "_dimension_" << N << "_recursions_" << R << "_range_" << range;
    return ss.str();
}

// Test cases (N, R, range). eps is used for R = 0 only (the same cuBLAS GEMM as the golden result);
// for R >= 1 the tolerance is relative to the output, see caps_eps(). When golden() loads a stored
// result the comparison is exact-ish (1e-8).
struct caps_case_t
{
    int N, R, range;
    double eps;
};

const caps_case_t caps_cases[] = {
    {512, 0, 1, 1e-6},      {4 * 512, 0, 2, 1e-6}, // no recursion: the same cuBLAS GEMM as the golden result
    {512, 1, 1, 3e-1},      {512, 1, 2, 7e-1},
    {2 * 512, 1, 1, 5e-1},  {2 * 512, 1, 2, 2e-0},
    {4 * 512, 1, 1, 1e-0},  {4 * 512, 1, 2, 4e-0},
    {8 * 512, 2, 1, 4e-0},  {8 * 512, 2, 2, 2e+1},
    {16 * 512, 2, 1, 1e+1}, {16 * 512, 2, 2, 3e+1},
    {16 * 512, 3, 1, 2e+1}, {16 * 512, 3, 2, 6e+1},
    // N = k * 2^R that is not a power of two (k = 3 * 2^j; select_kernel() / gen/tuning_table.hpp entries)
    {12288, 1, 1, 1e+1},    {12288, 2, 1, 2e+1},
    {24576, 2, 1, 3e+1},    {24576, 3, 1, 4e+1},
};

/// The rows of caps() and dfsCaps(): every case with half accumulation in the leaf GEMM, the
/// recursive ones (R >= 1) also with f32 accumulation (the golden result stays cuBLAS with half
/// accumulation, so the tolerance of caps_eps() applies), and for dfsCaps() also the overlapped
/// mode (two buffer sets, the form step on a second stream, auto_form_sms() SMs left free)
void add_caps_rows(bool dfs)
{
    QTest::addColumn<int>("N");
    QTest::addColumn<int>("R");
    QTest::addColumn<int>("range");
    QTest::addColumn<double>("eps");
    QTest::addColumn<bool>("f32");
    QTest::addColumn<bool>("overlapped");
    for (const auto &c : caps_cases) {
        const auto add = [&](bool f32, bool overlapped) {
            const auto row = QString("N=%1,R=%2,range=%3%4%5")
                                 .arg(c.N)
                                 .arg(c.R)
                                 .arg(c.range)
                                 .arg(f32 ? ",f32" : "")
                                 .arg(overlapped ? ",overlapped" : "")
                                 .toLatin1();
            QTest::newRow(row.constData()) << c.N << c.R << c.range << c.eps << f32 << overlapped;
        };
        add(false, false);
        if (c.R >= 1) {
            add(true, false);
        }
        if (dfs) {
            add(false, true);
        }
    }
}

/// Golden result: stored file if present (then eps is tightened), otherwise cuBLAS.
cv::HMat golden(const std::string &test_name,
               cv::cuda::GpuMat &A_dev,
               cv::cuda::GpuMat &B_dev,
               cv::cuda::Stream &stream,
               double &eps)
{
    auto C = load_test_result(test_name);
    if (GEMM::regenerate || C.empty()) {
        cv::cuda::GpuMat C_dev(A_dev.size(), A_dev.type());
        cublas_gemm_t cgemm(stream);
        cgemm.gemm(A_dev, B_dev, C_dev);
        C_dev.download(C, stream);
        stream.waitForCompletion();
        C = to_float(C);
    } else {
        eps = 1e-8;
    }
    return C;
}

/// Tolerance of a CAPS result (R >= 1) against the cuBLAS golden C, relative to the size of the
/// output instead of a fixed number per case: the error of half-precision Strassen-Winograd grows
/// with the magnitude of C, the accumulation length and the recursion depth, and with fixed numbers
/// new random data (a different test order) tipped borderline cases over.
///
/// unit = rms(C) * 2^-10 (about one half-precision step of a typical output value). Measured on the
/// RTX 4090, max |error| in units: R = 1, N = 512 / 1024 / 2048: 31 / 41 / 55; R = 2, N = 4096 / 8192:
/// 144 / 265; R = 3, N = 8192: 453 (the same for range 1 and 2). The bound
/// 3^R * sqrt(N) units leaves a 2 .. 5x margin; caps_mean_ok() checks the mean error against a tenth
/// of it, which catches a wrong block even where the max bound is loose.
/// eps_table is returned unchanged for R = 0 and for a stored golden result (golden() set 1e-8).
double caps_eps(const cv::Mat &gold, int N, int R, double eps_table)
{
    if (R <= 0 || eps_table <= 1e-8) {
        return eps_table;
    }
    const auto rms = cv::norm(gold, cv::NORM_L2) / std::sqrt(double(gold.total()));
    auto levels = 1.0; // 3^R
    for (auto r = 0; r < R; ++r) {
        levels *= 3;
    }
    return levels * std::sqrt(double(N)) * rms / 1024.0;
}

/// Mean |out - gold| <= eps / 10 (see caps_eps()); prints the numbers when it fails
bool caps_mean_ok(const std::string &name, const cv::Mat &gold, const cv::Mat &out, double eps)
{
    cv::Mat diff;
    cv::absdiff(gold, out, diff);
    const auto mean = cv::mean(diff)[0];
    const auto ok = mean <= eps / 10;
    if (!ok) {
        std::cerr << name << ": mean error " << mean << " > " << eps / 10 << " (eps / 10)" << std::endl;
    }
    return ok;
}

} // namespace

#define REQUIRE_GPU()                                                                          \
    do {                                                                                       \
        if (!m_gpu) {                                                                          \
            QSKIP("no CUDA device");                                                           \
        }                                                                                      \
    } while (0)

/// Skips the row when a CAPS run of (N, R) does not fit into this GPU (BFSCaps::fits_device_memory())
/// instead of failing on the allocation
#define REQUIRE_MEMORY(N, R)                                                                   \
    do {                                                                                       \
        std::string why_;                                                                      \
        if (!BFSCaps::fits_device_memory(N, R, &why_)) {                                           \
            QSKIP(why_.c_str());                                                               \
        }                                                                                      \
    } while (0)

class UnitTests : public QObject
{
    Q_OBJECT

    bool m_gpu = false;

private slots:
    void initTestCase();
    void cleanupTestCase();

    // CPU: test-only encoders and the MKL-backed reference wrappers
    void referenceWrappersMatchMkl_data();
    void referenceWrappersMatchMkl();

    // GPU
    void hgemm();
    void caps_data() { add_caps_rows(false); }
    void caps();
    void wgmmaMatchesCublas_data();
    void wgmmaMatchesCublas();
    void dfsCaps_data() { add_caps_rows(true); }
    void dfsCaps();
};

void UnitTests::initTestCase()
{
    srand(1);
    print_build_config(std::cerr);
    m_gpu = has_cuda_device();
    if (m_gpu) {
        report_device_capabilities();
    } else {
        qWarning("No CUDA device found: GPU tests will be skipped");
    }
}

void UnitTests::cleanupTestCase()
{
    if (m_gpu) {
        cv::HMat::Release();
    }
}

// ------------------------------------------------------------------------------------------
// CPU
// ------------------------------------------------------------------------------------------

void UnitTests::referenceWrappersMatchMkl_data()
{
    QTest::addColumn<int>("N");
    QTest::addColumn<int>("R");
    QTest::addColumn<double>("eps");
    QTest::newRow("N=256,R=2") << 256 << 2 << 5e-1;
    QTest::newRow("N=32,R=3") << 32 << 3 << 5e-1;
    QTest::newRow("N=256,R=4") << 256 << 4 << 2.0;
}

void UnitTests::referenceWrappersMatchMkl()
{
    QFETCH(int, N);
    QFETCH(int, R);
    QFETCH(double, eps);

    cv::Mat A(N, N, CV_32FC1), B(N, N, CV_32FC1), C_gold;
    cv::randu(A, cv::Scalar(-5), cv::Scalar(5));
    cv::randu(B, cv::Scalar(-5), cv::Scalar(5));
    cv::sgemm(A, B, C_gold);

    const auto T = N >> R;
    encode(T, A);
    encode(T, B);

    // layered reference on half-rounded inputs (what the GPU sees)
    std::map<int, std::vector<test_node_t>> nodes;
    calculate_layers(R, to_half(A), to_half(B), nodes);
    auto &C_layers = nodes[0].front().m_C;
    decode(T, C_layers);

    // serial reference in float with MKL leaves
    const cv::Mat B_before = B.clone();
    mult_opt(T, A, B);
    QVERIFY2(0 == cv::norm(B, B_before, cv::NORM_INF), "mult_opt must not modify B");
    decode(T, A);

    const auto name = std::string(__FUNCTION__) + "_" + std::to_string(N) + "_" + std::to_string(R);
    QVERIFY(calc_diff(C_gold, C_layers, name + "_mkl_vs_layers", eps));
    QVERIFY(calc_diff(C_gold, A, name + "_mkl_vs_serial", eps));
}

// ------------------------------------------------------------------------------------------
// GPU
// ------------------------------------------------------------------------------------------

void UnitTests::hgemm()
{
    REQUIRE_GPU();
    const auto N = 128;
    cv::cuda::Stream stream;
    cv::cuda::GpuMat A_dev, B_dev, C_dev;

    auto [AA, BB] = generate_half_data(N, 5, 0, true, false);
    cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
    cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);

    cv::Mat C_gold;
    cv::sgemm(to_float(AA), to_float(BB), C_gold);

    cublas_gemm_t cgemm(stream);
    cgemm.gemm(A_dev, B_dev, C_dev);

    cv::Mat H_out;
    C_dev.download(H_out, stream);
    stream.waitForCompletion();
    QVERIFY(calc_diff(C_gold, to_float(H_out), "hgemm_mkl_float_vs_cublas_half", 6e-1));
}

void UnitTests::caps()
{
    REQUIRE_GPU();
    QFETCH(int, N);
    QFETCH(int, R);
    QFETCH(int, range);
    QFETCH(double, eps);
    QFETCH(bool, f32);
    REQUIRE_MEMORY(N, R);

    // own name per variant: a stored result (golden()) is that of the same variant
    const auto test_name = name_of(f32 ? "test_caps_f32" : "test_caps", N, R, range);
    cv::cuda::Stream stream;
    auto [AA, BB] = generate_half_data(test_name, N, range, GEMM::int_range, GEMM::regenerate, false);

    cv::cuda::GpuMat A_dev, B_dev;
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);
    const auto C = golden(test_name, A_dev, B_dev, stream, eps);

    BFSCaps bfs;
    bfs.initialize(A_dev, B_dev, R, stream);
    bfs.forward_pass(stream);
    bfs.batched_leaf(stream, f32 ? wgmma_gemm_t::accumulator_t::single : wgmma_gemm_t::accumulator_t::half);
    bfs.backward_pass(stream);

    cv::HMat C3;
    bfs.get_result().download(C3, stream);
    stream.waitForCompletion();
    C3 = to_float(C3);
    eps = caps_eps(C, N, R, eps);
    QVERIFY(compare(test_name, AA, BB, C, C3, GEMM::regenerate, eps));
    QVERIFY(caps_mean_ok(test_name, C, C3, eps));
}

namespace {

/// Half data packed as CV_32FC1 (rows x cols/2 floats) -> CV_32F (rows x cols)
cv::Mat halves_to_float(const cv::Mat &packed)
{
    cv::Mat h(packed.rows, packed.cols * 2, CV_16FC1, const_cast<uchar *>(packed.data), packed.step), f;
    h.convertTo(f, CV_32F);
    return f;
}

/// First element where a and b (CV_32F) differ, as text; empty if equal
std::string first_difference(const cv::Mat &a, const cv::Mat &b)
{
    for (auto r = 0; r < a.rows; ++r) {
        for (auto c = 0; c < a.cols; ++c) {
            const auto x = a.at<float>(r, c), y = b.at<float>(r, c);
            if (!(x == y)) {
                return cv::format("(%d, %d): %g vs %g", r, c, x, y);
            }
        }
    }
    return {};
}

} // namespace

// Our mma.sync leaf GEMM with half accumulation (default configuration): the same pipeline and
// tolerances as caps(), which uses cuBLAS with half accumulation
void UnitTests::wgmmaMatchesCublas_data()
{
    QTest::addColumn<int>("N");
    QTest::addColumn<int>("R");
    QTest::newRow("N=1024,R=2") << 1024 << 2; // leaf 256: one 128 x 256 tile column, 49 products
    QTest::newRow("N=2048,R=1") << 2048 << 1; // leaf 1024
    QTest::newRow("N=4096,R=3") << 4096 << 3; // leaf 512, 343 products
    QTest::newRow("N=8192,R=2") << 8192 << 2; // leaf 2048
}

// Our Hopper leaf GEMM (wgmma_gemm_t: TMA + wgmma) vs cuBLAS on the leaf products of a real
// forward pass, whatever CMake LEAF_GEMM is (the plan is made here, not by BFS). Skipped unless
// the GPU has compute capability 9.0 and the build contains sm_90a code.
//   single accumulation: integer data, so every leaf product must be
//       bit-identical to cuBLAS's CUBLAS_COMPUTE_32F result (a difference = layout / descriptor bug);
//   half accumulation: rounding differs from cuBLAS's 16F order, so only the relative L2 error of
//       every product is checked.
void UnitTests::wgmmaMatchesCublas()
{
    REQUIRE_GPU();
    std::string why;
    if (!wgmma_gemm_t::available(why)) {
        QSKIP(why.c_str());
    }
    QFETCH(int, N);
    QFETCH(int, R);
    REQUIRE_MEMORY(N, R);

    cv::cuda::Stream stream;
    auto [AA, BB] = generate_half_data(N, 2, true, true, false); // integers in [-2, 1]
    cv::cuda::GpuMat A_dev, B_dev;
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);
    BFSCaps bfs;
    QVERIFY(bfs.initialize(A_dev, B_dev, R, stream));
    bfs.forward_pass(stream); // the operands (A, B quadrants, S areas) stay as they are below

    auto &buffers = bfs.m_mult_buffers;
    const auto leaf = N >> R;
    const auto plan = wgmma_gemm_t::make_plan(buffers.AA, buffers.ld_ab, buffers.BB, buffers.ld_ab, leaf);
    wgmma_gemm_t wgmma;
    wgmma.set_stream(stream);

    // the products are dense leaf x leaf buffers (ld_c = leaf): host copies of their addresses, to
    // poison them between the cuBLAS and the wgmma run
    const thrust::host_vector<half *> products = buffers.CC;
    CV_Assert(buffers.ld_c == leaf);
    const auto compare = [&](wgmma_gemm_t::accumulator_t accumulator, const QString &what) {
        std::vector<std::vector<cv::Mat>> expected, actual;
        bfs.batched_cublas_ex(stream, CUBLAS_GEMM_DFALT_TENSOR_OP,
                              accumulator == wgmma_gemm_t::accumulator_t::single ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F);
        bfs.download_leaf_mults(expected, stream);
        // every byte 0xFF (half NaN) before wgmma: a tile it does not write stays NaN and fails
        // (otherwise cuBLAS's correct values would still be there)
        for (const auto p : products) {
            cudaMemsetAsync(p, 0xFF, size_t(leaf) * size_t(leaf) * sizeof(half), (cudaStream_t) stream.cudaPtr());
        }
        wgmma.gemm(plan, buffers.AA, buffers.CC, buffers.ld_c, accumulator);
        bfs.download_leaf_mults(actual, stream);
        stream.waitForCompletion();
        QCOMPARE(actual.size(), expected.size());
        for (size_t n = 0; n < expected.size(); ++n) {
            for (size_t p = 0; p < expected[n].size(); ++p) {
                const auto e = halves_to_float(expected[n][p]);
                const auto a = halves_to_float(actual[n][p]);
                if (accumulator == wgmma_gemm_t::accumulator_t::single) {
                    const auto d = first_difference(e, a);
                    QVERIFY2(d.empty(), qPrintable(QString("single, %1: node %2 product %3 %4 (cuBLAS vs wgmma)")
                                                       .arg(what).arg(n).arg(p).arg(QString::fromStdString(d))));
                } else {
                    const auto norm = std::max(cv::norm(e, cv::NORM_L2), 1.0);
                    const auto rel = cv::norm(e, a, cv::NORM_L2) / norm;
                    QVERIFY2(rel < 1e-2, qPrintable(QString("half, %1: node %2 product %3 relative L2 error %4")
                                                        .arg(what).arg(n).arg(p).arg(rel)));
                }
            }
        }
    };
    // f32: bit-exact for every cluster size and schedule (each maps blocks to tiles differently);
    // f16: with the defaults
    using schedule_t = wgmma_gemm_t::schedule_t;
    for (const auto cluster : {1, 2}) {
        for (const auto schedule : {schedule_t::persistent, schedule_t::one_tile_per_block}) {
            wgmma.set_cluster(cluster);
            wgmma.set_schedule(schedule);
            compare(wgmma_gemm_t::accumulator_t::single,
                    QString("cluster %1, %2").arg(cluster).arg(schedule == schedule_t::persistent ? "persistent" : "one tile per block"));
        }
    }
    // the round barrier (set_round_sync()): persistent, cluster 1; the last round is partial here
    // (e.g. 343 leaves of 512: 2744 tiles on 132 CTAs), which exercises the early arrivals
    wgmma.set_cluster(1);
    wgmma.set_schedule(schedule_t::persistent);
    wgmma.set_round_sync(true);
    compare(wgmma_gemm_t::accumulator_t::single, "cluster 1, persistent, round sync");
    wgmma.set_round_sync(wgmma_gemm_t::DEFAULT_ROUND_SYNC);
    // the direct store of C (set_staged_store(false)); the cases above use the staged store
    wgmma.set_staged_store(false);
    compare(wgmma_gemm_t::accumulator_t::single, "cluster 1, persistent, direct store");
    compare(wgmma_gemm_t::accumulator_t::half, "cluster 1, persistent, direct store");
    wgmma.set_staged_store(wgmma_gemm_t::DEFAULT_STAGED_STORE);
    // 128 x 128 tiles forced (set_tile_n(128), cluster 1; automatic only for small batches), both
    // schedules (persistent falls back to one tile per block below 6 k slices, leaf 256)
    wgmma.set_tile_n(128);
    for (const auto schedule : {schedule_t::persistent, schedule_t::one_tile_per_block}) {
        wgmma.set_schedule(schedule);
        const auto what = QString("cluster 1, tile 128, %1").arg(schedule == schedule_t::persistent ? "persistent" : "one tile per block");
        compare(wgmma_gemm_t::accumulator_t::single, what);
        compare(wgmma_gemm_t::accumulator_t::half, what);
    }
    wgmma.set_tile_n(wgmma_gemm_t::DEFAULT_TILE_N);
    wgmma.set_cluster(wgmma_gemm_t::DEFAULT_CLUSTER);
    wgmma.set_schedule(wgmma_gemm_t::DEFAULT_SCHEDULE);
    compare(wgmma_gemm_t::accumulator_t::half, "defaults");
}

/// DFSCaps (wgmma builds, R >= 2) against the same golden result and bounds as caps()
void UnitTests::dfsCaps()
{
    REQUIRE_GPU();
    QFETCH(int, N);
    QFETCH(int, R);
    QFETCH(int, range);
    QFETCH(double, eps);
    QFETCH(bool, f32);
    QFETCH(bool, overlapped);
#if (LEAF_GEMM_WGMMA)
    if (R < 2) {
        QSKIP("DFSCaps needs R >= 2");
    }
    REQUIRE_MEMORY(N, R);

    const auto test_name = name_of((std::string("test_dfs_caps") + (f32 ? "_f32" : "") + (overlapped ? "_overlapped" : "")).c_str(), N, R, range);
    cv::cuda::Stream stream;
    auto [AA, BB] = generate_half_data(test_name, N, range, GEMM::int_range, GEMM::regenerate, false);

    cv::cuda::GpuMat A_dev, B_dev;
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);
    const auto C = golden(test_name, A_dev, B_dev, stream, eps);

    DFSCaps dfs;
    std::string why;
    if (!dfs.initialize(A_dev, B_dev, R, stream, &why, overlapped ? DFSCaps::mode_t::overlapped : DFSCaps::mode_t::serial)) {
        QSKIP(("DFSCaps: " + why).c_str());
    }
    dfs.run(stream, f32 ? wgmma_gemm_t::accumulator_t::single : wgmma_gemm_t::accumulator_t::half);

    cv::HMat C_out;
    dfs.get_result().download(C_out, stream);
    stream.waitForCompletion();
    C_out = to_float(C_out);
    eps = caps_eps(C, N, R, eps);
    QVERIFY(compare(test_name, AA, BB, C, C_out, GEMM::regenerate, eps));
    QVERIFY(caps_mean_ok(test_name, C, C_out, eps));
#else
    Q_UNUSED(N);
    Q_UNUSED(R);
    Q_UNUSED(range);
    Q_UNUSED(eps);
    Q_UNUSED(f32);
    Q_UNUSED(overlapped);
    QSKIP("DFSCaps needs LEAF_GEMM=wgmma");
#endif
}

QTEST_MAIN(UnitTests)
#include "unit_tests.moc"
