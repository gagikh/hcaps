#include <common.hpp>
#include <perf.hpp>
#include <map>
#include "test_utils.hpp"

#include <bfs_caps.hpp>
#include <core.hpp>
#include <hmat.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <cublas_gemm.hpp>

#include <iostream>

namespace GEMM {

// Performance test for CUBLAS vs MKL
static auto cublas_vs_mkl_performance_test(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs)
{
    cv::cuda::Stream stream;
    for (auto range = 1; range <= GEMM::max_range; ++range) {
        for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
            if (!GEMM::inputs_fit(N)) {
                break;
            }
            std::cerr << __FUNCTION__ << ", generate data for dimension=" << N
                      << ", range=" << range << std::endl;

            perf_t perf(__FUNCTION__);
            perf.start();
            // type, range, recursions, dimension;
            auto [AA, BB] = generate_half_data(N, range, GEMM::int_range, GEMM::regenerate, GEMM::keep, stream);
            auto A = to_float(AA, stream);
            auto B = to_float(BB, stream);

            cv::Mat C_mkl;
            {
                cv::HMat HA, HB;
                A.download(HA, stream);
                B.download(HB, stream);
                stream.waitForCompletion();
                cv::sgemm(HA, HB, C_mkl);
            }

            cv::cuda::GpuMat C_dev;
            cublas_gemm_t cgemm(stream);
            cgemm.gemm(AA, BB, C_dev);
            {
                C_dev = to_float(C_dev, stream);
                cv::HMat C_cublas;
                C_dev.download(C_cublas, stream);
                stream.waitForCompletion();

                auto stat = diff(C_mkl, C_cublas);
                for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                    statistcs[GEMM::PRECISION][range][R][N] = stat;
                }
            }

            {
                cuperf_t perf(__FUNCTION__);
                GEMM::run_test(perf, [&](cv::cuda::Stream &stream) { cgemm.gemm(AA, BB, C_dev); }, stream, measure);

                {
                    stat_t stat;
                    stat.maxVal = perf.get_max();
                    stat.avgVal = perf.get_avg();
                    stat.minVal = perf.get_min();
                    stat.stdVal = 0;
                    stat.relVal = 0;
                    for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                        statistcs[GEMM::PERFORMACE][range][R][N] = stat;
                    }
                }
            }
            perf.end();
            std::cerr << perf;
        }
    }
}

// Performance test for CAPS+CUBLAS vs MKL
static auto caps_cublas_vs_mkl_performance_test(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs)
{
    cv::cuda::Stream stream;
    for (auto range = 1; range <= GEMM::max_range; ++range) {
        for (auto N = 256; N < GEMM::max_dimension; N *= 2) {
            if (!GEMM::inputs_fit(N)) {
                break;
            }
            std::cerr << __FUNCTION__ << ", generate data for dimension=" << N
                      << ", range=" << range << std::endl;
            auto [AA, BB] = generate_half_data(N, range, GEMM::int_range, GEMM::regenerate, GEMM::keep, stream);

            cv::Mat C_mkl;
            {
                auto A = to_float(AA, stream);
                auto B = to_float(BB, stream);
                cv::HMat HA, HB;
                A.download(HA, stream);
                B.download(HB, stream);
                stream.waitForCompletion();
                cv::sgemm(HA, HB, C_mkl);
            }

            for (auto R = 1; R <= GEMM::max_recursions; ++R) {
                if (!GEMM::is_valid(N, R)) {
                    continue;
                }
                //std::cerr << "------>, R=" << R << std::endl;

                BFSCaps bfs;
                bfs.initialize(AA.clone(), BB.clone(), R, stream);
                bfs.forward_pass(stream);
                bfs.batched_leaf(stream, wgmma_gemm_t::accumulator_t::half);
                bfs.backward_pass(stream);

                {
                    auto C3 = to_float(bfs.get_result(), stream);
                    cv::HMat C4;
                    C3.download(C4, stream);
                    stream.waitForCompletion();
                    const auto stat = diff(C_mkl, C4);
                    statistcs[GEMM::PRECISION][range][R][N] = stat;
                }

                {
                    cuperf_t perf(__FUNCTION__);
                    GEMM::run_test(perf, [&](cv::cuda::Stream &stream) {
                        bfs.forward_pass(stream);
                        bfs.batched_leaf(stream, wgmma_gemm_t::accumulator_t::half);
                        bfs.backward_pass(stream);
                    }, stream, measure);
                    {
                        stat_t stat;
                        stat.maxVal = perf.get_max();
                        stat.avgVal = perf.get_avg();
                        stat.minVal = perf.get_min();
                        stat.stdVal = 0;
                        stat.relVal = 0;
                        statistcs[GEMM::PERFORMACE][range][R][N] = stat;
                    }
                }
            }
        }
    }
}

// Public interface functions
void run_cublas_performance_tests(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs)
{
    cublas_vs_mkl_performance_test(statistcs);
}

void run_caps_cublas_performance_tests(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs)
{
    caps_cublas_vs_mkl_performance_test(statistcs);
}

} // namespace GEMM
