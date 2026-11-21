/**
 * @file common.cpp
 * @brief Host implementations of the shared helpers declared in common.hpp
 * @author GEMM Project
 * @date 2025
 * 
 * Contents, in file order:
 *  - arch(), print_build_config(), report_device_capabilities(): what GPU and which build;
 *  - cv::sgemm(): MKL reference GEMM;
 *  - compare(): golden-result handling for the tests;
 *  - to_half() / to_float() on cv::Mat: conversion to / from the half-packed layout
 *    (N x M halves stored as an N x M/2 CV_32FC1 matrix, two halves per float slot);
 *  - store() / load(): raw binary matrix files in ../data/;
 *  - generate_half_data(): random test operands, cached in ../data/;
 *  - diff(), calc_diff(): error statistics;
 *  - get_SM_count(), cooperative_launch_config(), dump_best_config(), cooperative_launch():
 *    cooperative kernel launch support for the BFS kernels.
 * The device-side counterparts (urand, GpuMat to_half / to_float) are in test_utils.cu.
 */

#include <iostream>
#include <cuda_runtime_api.h>

#include <core.hpp>
#include <common.hpp>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <opencv2/cudaarithm.hpp>

#include <mkl_cblas.h>
#include <opencv2/core/mat.hpp>

#include <fstream>
#include <climits> // INT_MAX
#include <cstdlib>
#include <hmat.hpp>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector_types.h>

/**
 * @brief Get CUDA architecture name from compute capability
 * @return Architecture name, empty string if unknown
 * 
 * Maps the major compute capability to a human-readable name (Pascal 6.x to Blackwell 12.x)
 * and prints the capability to std::cerr. Only @p major is used, so 7.0 / 7.2 (Volta) are
 * also reported as "Turing", and 8.9 (Ada) as "Ampere"; 10.x is the data-center Blackwell
 * (B200), 12.x the consumer / workstation Blackwell.
 */
std::string arch(int major, int minor)
{
    auto name = "";
    std::cerr << "Arch compute=" << major << "." << minor << std::endl;
    if (6 == major) {
        name = "Pascal";
    } else if (7 == major) {
        name = "Turing";
    } else if (8 == major) {
        name = "Ampere";
    } else if (9 == major) {
        name = "Hopper";
    } else if (10 == major) {
        name = "Blackwell B200";
    } else if (12 == major) {
        name = "Blackwell";
    } else {
        std::cerr << "Unknow arch with compute=" << major << "." << minor << std::endl;
    }
    return name;
}

// Two-level stringification: the outer macro expands its argument first, so
// GEMM_STRINGIFY(GEMM_BUILD_CONFIG) yields the value of the macro (e.g. "Release"),
// not the text "GEMM_BUILD_CONFIG".
#define GEMM_STRINGIFY_(x) #x
#define GEMM_STRINGIFY(x) GEMM_STRINGIFY_(x)

/**
 * @brief Print the CMake options this binary was compiled with and what each one means
 *
 * GEMM_BUILD_CONFIG and GEMM_CUDA_ARCH are passed by CMake as compile definitions; the other
 * names (LAST_USE_HINTS, ...) are compile-time booleans from core.hpp, so the
 * report always matches the code that was actually compiled.
 */
void print_build_config(std::ostream &out)
{
#ifdef GEMM_BUILD_CONFIG
    const char *config = GEMM_STRINGIFY(GEMM_BUILD_CONFIG);
#else
    const char *config = "unknown";
#endif
#ifdef GEMM_CUDA_ARCH
    const char *cuda_arch = GEMM_STRINGIFY(GEMM_CUDA_ARCH);
#else
    const char *cuda_arch = "unknown";
#endif
#ifdef NDEBUG
    const char *asserts = "asserts off";
#else
    const char *asserts = "asserts on";
#endif
    const auto on = [](bool v) { return v ? "ON " : "OFF"; };
    const char *width = "half8 (16 B per thread access)"; // vector width of the element-wise passes
    out << "=== gemm build configuration (CMake options) ===\n"
        << "  build                         " << config << ", " << asserts << ", sm_" << cuda_arch
        << (PRODUCTION_CODE ? ", PRODUCTION_CODE (benchmark settings forced)" : "") << "\n"
        << "  LOOPS_COUNT_AS_TEMPLATE_ARGS  " << on(LOOPS_COUNT_AS_TEMPLATE_ARGS)
        << (LOOPS_COUNT_AS_TEMPLATE_ARGS ? "  row loop counts compiled in from gen/tuning_table.hpp"
                                         : "  row loop counts are kernel arguments (tuning build)") << "\n"
        << "  LAST_USE_HINTS                " << on(LAST_USE_HINTS)
        << (LAST_USE_HINTS ? "  last accesses of encode / decode / forward sums as __ldlu / __stcs"
                           : "  passes load / store with the L2 cache policy only") << "\n"
        << "  LEAF_GEMM                     " << (LEAF_GEMM_WGMMA ? "wgmma " : "cublas")
        << (LEAF_GEMM_WGMMA ? "  leaves: our TMA + wgmma GEMM (Hopper), R = 0 on cuBLAS" : "  leaves: batched cublasGemmEx")
        << "\n"
        << "  BFS_WARPS_Y                   " << BFS_WARPS_Y << (BFS_WARPS_Y < 10 ? "    " : "   ")
        << "blocks of 32 x " << BFS_WARPS_Y << " threads in the BFS passes" << "\n"
        << "  L2_CACHE_POLICY               " << on(L2_CACHE_POLICY)
        << (L2_CACHE_POLICY ? "  passes keep data in L2 (__ldcg / __stcg)" : "  passes stream (__ldcs / __stcs)")
        << "\n"
        << "  FULL_UNROLL_ROW_LOOPS         " << on(FULL_UNROLL_ROW_LOOPS)
        << (FULL_UNROLL_ROW_LOOPS ? "  row loops fully unrolled (slow compile)" : "  row loops not unrolled") << "\n"
        << "  vector width                  " << width << "\n"
        << "  REPORT_STATISTICS             " << on(REPORT_STATISTICS)
        << (REPORT_STATISTICS ? "  launch statistics printed (occupancy queries in the launch path)" : "") << "\n"
        << "  USE_NVTX                      " << on(USE_NVTX) << (USE_NVTX ? "  NVTX ranges for the profiler" : "")
        << "\n"
        << "=================================================" << std::endl;
}

/**
 * @brief Print the properties of CUDA device 0 to std::cerr
 * @return Device name string
 * 
 * Prints architecture, cache and shared memory sizes, register and thread limits, clocks,
 * tensor core count, free / total memory and the theoretical DRAM bandwidth. Used at the
 * start of tests and benchmarks so that logs record the hardware they ran on.
 * 
 * @note Always reports device 0 even if several devices exist. cudaMemGetInfo needs a
 *       context, so calling this function also initializes CUDA on device 0.
 * @see arch() for architecture name mapping
 */
std::string report_device_capabilities()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cerr << "Total CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // CUDA 13 removed clockRate, memoryClockRate, computeMode and singleToDoublePrecisionPerfRatio
    // from cudaDeviceProp; the device attributes give the same values in CUDA 12 and 13
    const auto attribute = [](cudaDeviceAttr a) {
        int value = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&value, a, 0));
        return value;
    };
    std::cerr << "Using device: " << 0 << " of " << deviceCount << std::endl;
    std::cerr << "Device name: " << prop.name << std::endl;
    
    const auto name = arch(prop.major, prop.minor);
    std::cerr << "Architecture " << name << std::endl;

    // L2 size in bytes, printed in human-readable units by UBytes() (core.hpp)
    std::cerr << "L2 Cache Size: " << UBytes(prop.l2CacheSize) << std::endl;

    int sharedMemoryPerSM;
    CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    // the maximum shared memory per SM (carved out of the unified L1 / shared memory array),
    // not the full L1 size despite the label
    std::cerr << "L1 Cache Size (Shared Memory per SM): " << UBytes(sharedMemoryPerSM) << std::endl;
    std::cerr << "CUDA shared mem per block  " << UBytes(prop.sharedMemPerBlock) << std::endl;
    std::cerr << "CUDA registers per block  " << prop.regsPerBlock << std::endl;
    std::cerr << "CUDA registers per SM  " << prop.regsPerMultiprocessor << std::endl;
    
    // Compute capabilities
    std::cerr << "Warp size: " << prop.warpSize << std::endl;
    std::cerr << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cerr << "Max block dimensions: [" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cerr << "Max grid dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cerr << "CUDA total const memory " << UBytes(prop.totalConstMem) << std::endl;
    std::cerr << "CUDA compute mode: " << attribute(cudaDevAttrComputeMode) << std::endl;
    std::cerr << "CUDA async engines : " << prop.asyncEngineCount << std::endl;
    std::cerr << "CUDA persist cache size: " << UBytes(prop.persistingL2CacheMaxSize) << std::endl;
    std::cerr << "CUDA max threads per mproc " << prop.maxThreadsPerMultiProcessor << std::endl;
    // 32 = warp size on all supported architectures
    std::cerr << "CUDA max warps per mproc " << prop.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cerr << "CUDA global L1 " << (prop.globalL1CacheSupported ? "Yes" : "No") << std::endl;
    std::cerr << "CUDA local L1 " << (prop.localL1CacheSupported ? "Yes" : "No") << std::endl;
    std::cerr << "CUDA float/double perf " << attribute(cudaDevAttrSingleToDoublePrecisionPerfRatio) << std::endl;
    // clockRate and memoryClockRate are reported in kHz
    std::cerr << "CUDA clock rate " << attribute(cudaDevAttrClockRate) / 1000.0 << "MHz" << std::endl;
    std::cerr << "CUDA memory clock rate " << attribute(cudaDevAttrMemoryClockRate) / 1000.0 << "MHz" << std::endl;
    std::cerr << "CUDA bus width " << prop.memoryBusWidth / 8 << " bytes" << std::endl;
    std::cerr << "CUDA reserved shared mem " << UBytes(prop.reservedSharedMemPerBlock) << std::endl;
    std::cerr << "CUDA concurrent kernels " << prop.concurrentKernels << std::endl;
    
    // Advanced features
    std::cerr << "ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cerr << "Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    
    int streamPriorities;
    CUDA_CHECK(cudaDeviceGetAttribute(&streamPriorities, cudaDevAttrStreamPrioritiesSupported, 0));
    std::cerr << "Stream priorities supported: " << (streamPriorities ? "Yes" : "No") << std::endl;

    //cudaDeviceSetLimit(); (placeholder, no device limits are changed)
    std::cerr << "Using " << prop.multiProcessorCount << " MP on " << prop.major << "."
              << prop.minor << std::endl;
    // Tensor Core specific info. Not queryable through the runtime API, so the per-SM count
    // is hard-coded from NVIDIA's architecture documentation; unknown 7.x+ versions print nothing.
    int tensorCoresPerSM = 0;
    if (prop.major >= 7) {  // Volta and newer
        if (prop.major == 7 && prop.minor == 0) tensorCoresPerSM = 8;      // Volta
        else if (prop.major == 7 && prop.minor == 5) tensorCoresPerSM = 8;  // Turing
        else if (prop.major == 8) tensorCoresPerSM = 4;                     // Ampere
        else if (prop.major == 9) tensorCoresPerSM = 4;                     // Hopper
        else if (prop.major == 12) tensorCoresPerSM = 4;                    // Blackwell
        
        if (tensorCoresPerSM > 0) {
            std::cerr << "Tensor cores per SM: " << tensorCoresPerSM << std::endl;
            std::cerr << "Total tensor cores: " << prop.multiProcessorCount * tensorCoresPerSM << std::endl;
        }
    } else {
        // NOTE: architectures before Volta have no tensor cores; this branch prints a
        // meaningless number (SM count * 16) and is only reached on Pascal.
        std::cerr << "Total tensor cores: " << prop.multiProcessorCount * 16 << std::endl;
    }

    size_t free_t, total_t;
    CUDA_CHECK(cudaMemGetInfo(&free_t, &total_t));
    std::cerr << "Total GPU memory: " << UBytes(total_t) << std::endl;
    std::cerr << "Free GPU memory: " << UBytes(free_t) << std::endl;
    std::cerr << "Memory usage: " << UBytes(total_t - free_t) << " / " << UBytes(total_t) << std::endl;

    // Peak DRAM bandwidth = 2 (double data rate) * memory clock * bus width. Value in GB/s:
    // MHz * bytes = MB/s, / 1000 = GB/s. A theoretical upper bound, not a measurement.
    const auto memory_clock_rate_mhz = attribute(cudaDevAttrMemoryClockRate) / 1000.0;
    const auto bus_width_bytes = prop.memoryBusWidth / 8;
    const auto memory_bandwidth_gbps = 2.0 * memory_clock_rate_mhz * bus_width_bytes / 1000.0;
    std::cerr << "Theoretical memory bandwidth: " << memory_bandwidth_gbps << " GB/s" << std::endl;
    return prop.name;
}

namespace cv {

/**
 * @brief C = A * B, single precision, through MKL's cblas_sgemm (CPU reference for the tests)
 *
 * Operands are square N x N CV_32FC1. The leading dimensions are taken from the row steps
 * (step / sizeof(float)), so ROIs of a larger matrix work as inputs. alpha = 1, beta = 0.
 */
void sgemm(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
{
    CV_Assert(A.size() == B.size());
    CV_Assert(A.type() == B.type());
    C.create(A.size(), A.type());
    const auto N = A.rows;
    constexpr auto alpha = 1.0f;
    constexpr auto betta = 0.0f; // beta = 0: C is overwritten, its old content is ignored
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                N,
                N,
                N,
                alpha,
                A.ptr<float>(0),
                A.step / sizeof(float),
                B.ptr<float>(0),
                B.step / sizeof(float),
                betta,
                C.ptr<float>(0),
                C.step / sizeof(float));
}

}

/// Check @p out against @p gold (calc_diff with absolute tolerance @p eps). When the inputs were
/// regenerated and the check passed, store @p out as the new golden result (no dialog: the
/// tests also run headless, on compute nodes without a display).
/// AA and BB are not used.
bool compare(const cv::String &test_name,
             const cv::Mat &AA,
             const cv::Mat &BB,
             const cv::Mat &gold,
             const cv::Mat &out,
             bool regenerate,
             double eps)
{
    auto flag = calc_diff(gold, out, test_name, eps);
    if (flag && regenerate) {
        store_test_result(out, test_name);
        std::cerr << "Test result stored as the golden result: " << test_name << std::endl;
    }
    else if (!flag) {
        std::cerr << "Test results are not stored: " << test_name << std::endl;
    }
    return flag;
}

/**
 * @brief Float -> half-packed conversion on the host
 *
 * The N x M float matrix becomes an N x M/2 CV_32FC1 matrix (in pinned memory, so it can be
 * uploaded asynchronously); each float slot of the output stores two consecutive halves of
 * the row. Rounding is round-to-nearest-even (__float2half).
 */
cv::HMat to_half(const cv::Mat& A)
{
    const auto N = A.rows;
    const auto M = A.cols;
    CV_Assert(0 == (M & 1));
    CV_Assert(A.type() == CV_32FC1);
    cv::HMat AA(N, M / 2, CV_32FC1);

    for (auto i = 0; i < N; ++i) {
        // view output row i (M/2 floats) as M halves
        auto aa = reinterpret_cast<half*>(AA.ptr<float>(i));
        for (auto j = 0; j < M; ++j) {
            aa[j] = __float2half(A.at<float>(i, j));
        }
    }
    return AA;
}

/**
 * @brief Half-packed -> float conversion on the host (inverse of to_half(const cv::Mat&))
 *
 * Input is N x M CV_32FC1 holding N x 2M halves; output is an ordinary N x 2M float matrix.
 */
cv::Mat to_float(const cv::Mat& A)
{
    const auto N = A.rows;
    const auto M = A.cols;;
    CV_Assert(A.type() == CV_32FC1);
    cv::Mat AA(N, 2 * M, CV_32FC1);

    for (auto i = 0; i < N; ++i) {
        // view input row i (M floats) as 2M halves
        auto aa = reinterpret_cast<const half*>(A.ptr<float>(i));
        for (auto j = 0; j < 2 * M; ++j) {
            AA.at<float>(i, j) = __half2float(aa[j]);
        }
    }
    return AA;
}

/**
 * @brief Write @p mat to ../data/<path> as raw binary: int rows, int cols, int type, then the
 *        pixel data. The ".yaml" suffix used by callers is historical; the file is not YAML.
 *
 * The path is relative to the current working directory (normally the build directory, so the
 * files land in the repository's data/ folder). The matrix must be continuous, because the
 * data are written as one block of total() * elemSize() bytes. The format is native-endian and
 * not portable between platforms with different int size or byte order.
 */
void store(const cv::Mat &mat, const cv::String &path)
{
    std::ofstream ofs("../data/" + path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }

    // Write matrix metadata
    int rows = mat.rows;
    int cols = mat.cols;
    int type = mat.type();
    ofs.write((char*)&rows, sizeof(rows));
    ofs.write((char*)&cols, sizeof(cols));
    ofs.write((char*)&type, sizeof(type));

    // Write matrix data
    ofs.write((char*)mat.data, mat.total() * mat.elemSize());
}

/**
 * @brief Read a file written by store(). Returns an empty matrix if the file cannot be opened,
 *        which callers treat as "not cached yet". No validation of the header is done.
 */
cv::HMat load(const cv::String &path)
{
    std::ifstream ifs("../data/" + path, std::ios::binary);
    if (!ifs) {
        return cv::HMat();
    }

    // Read matrix metadata
    int rows, cols, type;
    ifs.read((char*)&rows, sizeof(rows));
    ifs.read((char*)&cols, sizeof(cols));
    ifs.read((char*)&type, sizeof(type));

    // Allocate matrix
    cv::HMat mat(rows, cols, type);

    // Read matrix data
    ifs.read((char*)mat.data, mat.total() * mat.elemSize());
    return mat;
}

/**
 * @brief Host test operands: uniform random N x N matrices in [-range, +range), converted to
 *        the half-packed layout. Cached as ../data/dimension_<N>_range_<range>_<int|float>_{A,B}.yaml.
 *
 * With int_range the values are drawn as CV_32SC1 (so they are integers) and converted to
 * float before the half conversion. If the cache cannot be loaded the function calls itself
 * with regenerate = true.
 */
std::pair<cv::HMat, cv::HMat> generate_half_data(
    int N, int range, bool int_range, bool regenerate, bool keep)
{
    std::stringstream ss;
    ss << "dimension_" << N << "_range_" << range << "_" << (int_range ? "int" : "float");
    return generate_half_data(ss.str(), N, range, int_range, regenerate, keep);
}

/// Same as above, with the cache file names given explicitly: ../data/<test_name>_{A,B}.yaml.
std::pair<cv::HMat, cv::HMat> generate_half_data(
    const std::string &test_name, int N, int range, bool int_range, bool regenerate, bool keep)
{
    if (regenerate) {
        const auto type = int_range ? CV_32SC1 : CV_32FC1;
        cv::Mat A(N, N, type);
        cv::Mat B(N, N, type);

        auto low = cv::Scalar(-range);
        auto high = cv::Scalar(+range);
        cv::randu(A, low, high);
        cv::randu(B, low, high);

        if (int_range) {
            A.convertTo(A, CV_32FC1);
            B.convertTo(B, CV_32FC1);
        }
        auto AA = to_half(A);
        auto BB = to_half(B);
        if (keep) {
            store(AA, test_name + "_A.yaml");
            store(BB, test_name + "_B.yaml");
        }
        return std::make_pair(AA, BB);
    }
    else {
        auto AA = load(test_name + "_A.yaml");
        auto BB = load(test_name + "_B.yaml");
        if (AA.empty() || BB.empty()) {
            return generate_half_data(test_name, N, range, int_range, true, keep);
        }
        return std::make_pair(AA, BB);
    }
}

/**
 * @brief Device test operands. New data are drawn on the GPU with urand() and converted with the
 *        GpuMat to_half(); cached data (same file names as the host overload) are uploaded.
 *
 * Seeds come from std::rand(), so the data depend on the srand() state of the program.
 */
std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> generate_half_data(
    int N, int range, bool int_range, bool regenerate, bool keep, cv::cuda::Stream &stream)
{
    std::stringstream ss;
    ss << "dimension_" << N << "_range_" << range << "_" << (int_range ? "int" : "float");
    auto test_name = ss.str();
    cv::cuda::GpuMat AA, BB;
    if (regenerate) {
        const auto type = int_range ? CV_32SC1 : CV_32FC1;
        cv::cuda::GpuMat A(N, N, type);
        cv::cuda::GpuMat B(N, N, type);
        // separate std::rand() seeds, otherwise A and B would be identical
        urand(range, std::rand(), A, stream);
        urand(range, std::rand(), B, stream);

        if (int_range) {
            A.convertTo(A, CV_32FC1, stream);
            B.convertTo(B, CV_32FC1, stream);
        }
        to_half(A, AA, stream);
        to_half(B, BB, stream);

        if (keep) {
            // download into pinned memory, then wait: store() reads the host buffers
            cv::HMat HA, HB;
            AA.download(HA, stream);
            BB.download(HB, stream); // was AA twice: B was stored as an empty matrix
            stream.waitForCompletion();
            store(HA, test_name + "_A.yaml");
            store(HB, test_name + "_B.yaml");
        }
        return std::make_pair(AA, BB);
    } else {
        auto HA = load(test_name + "_A.yaml");
        auto HB = load(test_name + "_B.yaml");
        if (HA.empty() || HB.empty()) {
            return generate_half_data(N, range, int_range, true, keep, stream);
        }
        AA.upload(HA, stream);
        BB.upload(HB, stream);
        stream.waitForCompletion(); // HA / HB are locals: the copies must finish before they are released
        return std::make_pair(AA, BB);
    }
}

/// Statistics of |A - B| on the host: min, max, mean, standard deviation and the relative L2
/// error ||A - B|| / (||B|| + 1e-5) with B as the reference. The 1e-5 avoids division by zero.
stat_t diff(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat diff;
    cv::absdiff(A, B, diff);
    cv::Scalar meanVal, stdVal;
    cv::meanStdDev(diff, meanVal, stdVal);

    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);

    constexpr auto eps = 1e-5;
    const auto norm_diff = cv::norm(diff, cv::NORM_L2);
    const auto norm_B = cv::norm(B, cv::NORM_L2);
    const auto relative_error = norm_diff / (norm_B + eps);

    stat_t stat;
    stat.minVal = minVal;
    stat.maxVal = maxVal;
    stat.avgVal = meanVal[0];
    stat.stdVal = stdVal[0];
    stat.relVal = relative_error;
    return stat;
}

/// Statistics of |A - B| on the device: only min, max and mean are computed (stdVal and relVal
/// are set to 0, see the disabled code at the end). Synchronizes @p stream twice.
stat_t diff(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::Stream &stream)
{
    stat_t stat;

    cv::cuda::GpuMat diff, loc, mean;

    cv::cuda::absdiff(A, B, diff, stream);
    // findMinMax writes a 1 x 2 matrix [min, max] of the element type
    cv::cuda::findMinMax(diff, loc, cv::noArray(), stream);
    // view diff as a single row (requires a continuous matrix) and average it into a 1 x 1 double
    cv::cuda::reduce(diff.reshape(1, 1), mean, 1, cv::REDUCE_AVG, CV_64FC1, stream);
    {
        cv::HMat min_max_loc_val;
        loc.download(min_max_loc_val, stream);
        stream.waitForCompletion();

        // convert [min, max] to double directly into vals (the Mat header wraps the array)
        double vals[2];
        min_max_loc_val.convertTo(cv::Mat(1, 2, CV_64FC1, &vals[0]), CV_64F);
        stat.minVal = vals[0];
        stat.maxVal = vals[1];
    }
    {
        cv::HMat mean_val;
        mean.download(mean_val, stream);
        stream.waitForCompletion();
        stat.avgVal = mean_val.at<double>(0, 0);
    }
    stat.relVal = 0;
    stat.stdVal = 0;
    return stat;
}

/// Absolute difference test: passes if max |A - B| < eps. Prints "<msg> passed|failed" with the
/// max, mean and standard deviation of |A - B| to std::cerr.
bool calc_diff(const cv::Mat &A, const cv::Mat &B, const std::string &msg, double eps)
{
    cv::Mat diff;
    cv::absdiff(A, B, diff);
    cv::Scalar meanVal, stdVal;
    cv::meanStdDev(diff, meanVal, stdVal);

    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);
    const auto flag = maxVal < eps;
    const auto status = flag ? "passed" : "failed";

    std::cerr << msg << " " << status << ", max=" << maxVal << ", mean=" << meanVal[0]
              << ", std=" << stdVal[0] << std::endl;
    return flag;
}

// See common.hpp: OpenCV's createContinuous() overflows its int element count from 2^31 elements.
void create_continuous(int rows, int cols, int type, cv::cuda::GpuMat &mat)
{
    const auto elements = size_t(rows) * size_t(cols);
    if (elements <= size_t(INT_MAX)) {
        cv::cuda::createContinuous(rows, cols, type, mat); // the usual path, unchanged
        return;
    }
    const auto row_bytes = size_t(cols) * CV_ELEM_SIZE(type);
    mat.create(rows, cols, type); // pitched allocation, size_t arithmetic
    if (mat.step != row_bytes) {
        CV_Error(cv::Error::StsNoMem, cv::format("create_continuous: %d x %d buffer got a pitch of %zu B instead of %zu B "
                                                 "(rows padded, not one continuous block)",
                                                 rows, cols, size_t(mat.step), row_bytes));
    }
}

/// Number of SMs of device 0 (the project assumes a single GPU).
int get_SM_count()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

/**
 * @brief Upper bound on the grid size of a cooperative launch of @p fcn with @p block threads:
 *        occupancy (blocks per SM with 0 bytes of dynamic shared memory) * SM count.
 */
int cooperative_launch_config(void *fcn, dim3 block, bool print)
{
    int numBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fcn, block.x * block.y * block.z, 0));
    static auto multiProcessorCount = get_SM_count(); // queried once and cached (single GPU)

    if (REPORT_STATISTICS && print) {
        std::cerr << "Number of feasible active blocks per SM: " << numBlocks << std::endl;
    }
    const auto totalActiveBlocks = numBlocks * multiProcessorCount;
    if (REPORT_STATISTICS && print) {
        std::cerr << "Total active blocks : " << totalActiveBlocks << std::endl;
    }
    return totalActiveBlocks;
}

/// Diagnostics (REPORT_STATISTICS only): occupancy-optimal block size and minimum grid size from
/// cudaOccupancyMaxPotentialBlockSize, plus the kernel's max threads per block and registers.
void dump_best_config(void *fcn)
{
    if (REPORT_STATISTICS) {
        int minGridSize, bestBlockSize;
        cudaError_t err;

        // Query optimal block size
        // no dynamic shared memory (0), no block size limit (0)
        err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, fcn, 0, 0);

        if (err == cudaSuccess) {
            std::cerr << "Best block size: " << bestBlockSize << std::endl;
            std::cerr << "Minimum grid size: " << minGridSize << std::endl;
        } else {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaFuncAttributes attr;
        err = cudaFuncGetAttributes(&attr, fcn);
        if (err == cudaSuccess) {
            std::cerr << "Max threads count: " << attr.maxThreadsPerBlock << std::endl;
            std::cerr << "Registers count: " << attr.numRegs << std::endl;
        }
    }
}

/**
 * @brief Cooperative launch on @p stream (needed by kernels that use grid.sync()).
 *
 * All blocks of a cooperative grid must be resident at once. If @p grid is larger than
 * cooperative_launch_config() allows, a message is printed but the launch is still issued:
 * cudaLaunchCooperativeKernel then fails (cudaErrorCooperativeLaunchTooLarge) and CUDA_CHECK
 * throws. No dynamic shared memory is passed.
 */
void cooperative_launch(void* fcn, void* args[], dim3 grid, dim3 block, cv::cuda::Stream& stream)
{
    const auto totalActiveBlocks = cooperative_launch_config(fcn, block, false);
    //std::cerr << "TOTAL: " << totalActiveBlocks << std::endl;
    const auto requestedBlocks = int(grid.x * grid.y * grid.z);
    if (totalActiveBlocks < requestedBlocks) {
        std::cerr << "Unable to launch cooperative group, total active blocks " << totalActiveBlocks << ", requested blocks "
                  << requestedBlocks << std::endl;
    } else {
        if (REPORT_STATISTICS && totalActiveBlocks != requestedBlocks) {
            std::cerr << "Total active blocks: " << totalActiveBlocks
                      << ", requested blocks: " << requestedBlocks << std::endl;
        }
    }

    CUDA_CHECK(cudaLaunchCooperativeKernel(fcn, grid, block, args, 0, (cudaStream_t) stream.cudaPtr()));
}
