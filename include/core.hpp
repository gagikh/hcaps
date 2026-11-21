/**
 * @file core.hpp
 * @brief Core definitions and configuration for GEMM project
 * @author GEMM Project
 * @date 2025
 * 
 * This file contains fundamental definitions, configuration options,
 * and utility macros used throughout the GEMM implementation.
 *
 * Included by nearly every translation unit (host .cpp and device .cu). It provides:
 *  - build options: macros set by CMake (option(...) in CMakeLists.txt, passed as -D...=0/1)
 *    with a fallback default here for builds outside CMake. A production build
 *    (PRODUCTION_CODE=ON) forces several of them to the benchmarked values;
 *  - algorithm constants: MULTS_COUNT (7 products per Strassen-Winograd node),
 *    MAX_RECURSIONS and the sizes of the constant-memory address caches derived from it;
 *  - the packed half vector types half8 used by the BFS kernels;
 *  - NVTX_RANGE() profiling markers;
 *  - uniform CUDA / cuBLAS error checking (CUDA_CHECK, CUDA_WARN, cublasSafeCall);
 *  - small host helpers (byte-count pretty printing).
 *
 * The option macros are tested with #if, so they must expand to an integer or true/false.
 */

#pragma once

#include <opencv2/core/base.hpp>
#include <iostream>
#include <string>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cublas.h> // cublasStatus_t; the project uses the legacy header everywhere (cublas_api.h may not be included directly, cublas_v2.h cannot be mixed with cublas.h)

/** @brief log2(WARP_SIZE): 32 == 1 << 5; currently not referenced by src/ or include/ */
#define WARP_ORDER 5
/** @brief CUDA warp size constant */
#define WARP_SIZE 32

/** @brief M = N = K of one WMMA / mma tile (16 x 16 x 16 half fragments). bfs_caps.cu requires leaf
 *  blocks (N >> R) of at least 4 * WMMA_MNK = 64 elements per side */
#define WMMA_MNK 16

// L2 cache policy is now defined by CMake
#ifndef L2_CACHE_POLICY
/** @brief Cache hints of the BFS passes (encode, sums, reconstruct, decode; see utils.cuh):
 *  true = keep the data in L2 (__ldcg / __stcg, the next pass reads it again),
 *  false = streaming accesses (__ldcs / __stcs) */
#define L2_CACHE_POLICY true
#endif


// LOOPS_COUNT_AS_TEMPLATE_ARGS is now defined by CMake
#ifndef LOOPS_COUNT_AS_TEMPLATE_ARGS
/** @brief true: the per-(N, R) loop counts of the BFS kernels (gen/tuning_table.hpp) are template
 *  arguments, so every tuned (N, R) has its own compiled kernel (select_kernel() in bfs_caps.cu);
 *  false: they are run-time arguments (one kernel for all), which the tuner needs to try
 *  arbitrary loop counts without recompiling */
#define LOOPS_COUNT_AS_TEMPLATE_ARGS true
#endif

/** @brief Products per Strassen-Winograd step (m1 .. m7): the branching factor of the recursion
 *  tree, so level l has 7^l nodes and R recursions give 7^R leaf products */
#define MULTS_COUNT 7
/** @brief Sums per internal node: S1, S2, S3 (from A quadrants) and S4, S5, S6 (from B quadrants),
 *  the operands of m3, m5, m6, m7 (see BFSCaps::sums_rows() for their layout). Also sizes BFS's
 *  per-sum stream / event arrays. The misspelling is part of the identifier and is kept. */
#define ARITHEMTIC_OPS_COUNT 6

// REPORT_STATISTICS is now defined by CMake
#ifndef REPORT_STATISTICS
/** @brief Print launch statistics (grid sizes, occupancy queries in the launch path) and CUDA graph
 *  status messages. Debug aid only: it adds host work to every launch */
#define REPORT_STATISTICS false
#endif

// USE_NVTX is now defined by CMake
#ifndef USE_NVTX
/** @brief Enable NVIDIA Tools Extension (NVTX) ranges (NVTX_RANGE) for Nsight Systems profiling */
#define USE_NVTX false
#endif

#ifndef LEAF_GEMM_WGMMA
/** @brief Leaf GEMM on Hopper: true = our TMA + wgmma kernel (include/wgmma_gemm.hpp, sm_90a only;
 * R = 0 runs on cuBLAS), false = batched cublasGemmEx. CMake LEAF_GEMM=wgmma / cublas */
#define LEAF_GEMM_WGMMA false
#endif

#ifndef BFS_WARPS_Y
/** @brief Warps along y in a block of the BFS passes (BFSCaps::WARPS_IN_Y_DIR; CMake BFS_WARPS_Y):
 * blocks of 32 x BFS_WARPS_Y threads, virtual blocks BFS_WARPS_Y rows high; 1, 2, 4, 8, 16 or 32 */
#define BFS_WARPS_Y 4
#endif

#ifndef LAST_USE_HINTS
/** @brief Last accesses of the encoder / decoder / forward sums as __ldlu / __stcs (src/utils.cuh fetch_with_policy).
 * Off: no gain at the tuned loop counts on the 4090 (DRAM traffic already minimal, CAPS within 0.4%) */
#define LAST_USE_HINTS false
#endif

#ifndef FULL_UNROLL_ROW_LOOPS
/** @brief Fully unroll the outer "row" loops of the BFS kernels (see GEMM_ROW_LOOP_UNROLL) */
#define FULL_UNROLL_ROW_LOOPS false
#endif

/**
 * @brief Unroll policy for the outer row loops (encode/decode, bilinear, reconstruct).
 * @details With LOOPS_COUNT_AS_TEMPLATE_ARGS their trip counts are compile-time constants
 * (up to 2048), and a full unroll copies bodies that each end in a grid-wide barrier.sync()
 * hundreds of times: ~6x longer compile (bfs_caps.cu 104 s -> 17 s with "unroll 1") for no expected
 * run-time gain. FULL_UNROLL_ROW_LOOPS=ON restores the old behaviour for benchmarking.
 */
#if FULL_UNROLL_ROW_LOOPS
#define GEMM_ROW_LOOP_UNROLL _Pragma("unroll")
#else
#define GEMM_ROW_LOOP_UNROLL _Pragma("unroll 1")
#endif

/**
 * @brief Half precision data types for optimized memory access
 * @details These types pack multiple half-precision values into wider types
 * for better memory bandwidth utilization and SIMD operations.
 * They are plain storage aliases: a half8 is the 16 bytes of 8 halves in a float4, so one load or
 * store moves 8 elements. The element-wise arithmetic (operator+ / operator- as 4 half2 ops) is
 * defined for half8 in src/utils.cuh. Matrices are N x N halves stored row-major; dimensions in
 * units of these types are divided by sizeof(T) / sizeof(half) ("DIM_SCALE" in the kernels).
 */
typedef float4 half8;  ///< Pack 8 half-precision values into float4

/**
 * @brief Number of recursions R (Strassen-Winograd levels)
 * @details R = 0 is a plain GEMM (no tree); R >= 1 gives 7^R leaf products of (N/2^R)^2 blocks.
 * The tree has levels 0 (root) .. R (leaves); the internal nodes are levels 0 .. R - 1.
 *
 * Constant-memory address caches (64 KB, 40-byte entries -> at most 1638 entries):
 * - internal nodes 1 + 7 + ... + 7^(R-1):   R = 1 -> 1, R = 2 -> 8, R = 3 -> 57, R = 4 -> 400
 * - parents of the leaves 7^(R-1):          R = 1 -> 1, R = 2 -> 7, R = 3 -> 49, R = 4 -> 343
 * so MAX_RECURSIONS = 4 (R = 5 would need 2801 entries); gen/ also has depth_1..4 functions only.
 */
#define MAX_RECURSIONS 4

/** @brief Maximum size for intermediate address cache (internal nodes for MAX_RECURSIONS) */
#define ADDRESS_CACHE_SIZE (0 + 1 + 7 + 7 * 7 + 7 * 7 * 7)
/** @brief Maximum size for multiplication address cache (parents of the leaves for MAX_RECURSIONS) */
#define MULT_CACHE_SIZE (7 * 7 * 7)

#if USE_NVTX
#include <nvtx3/nvToolsExt.h>

/**
 * @class NvtxTracer
 * @brief RAII wrapper for NVIDIA Tools Extension (NVTX) range markers
 * @details Automatically manages NVTX range push/pop operations for profiling
 */
class NvtxTracer {
public:
    /**
     * @brief Constructor for NVTX range with C-string name
     * @param name Range name as C-string
     */
    NvtxTracer(const char* name) {
        nvtxRangePushA(name);
    }
    /**
     * @brief Constructor for NVTX range with std::string name
     * @param name Range name as std::string
     */
    NvtxTracer(const std::string& name) {
        nvtxRangePushA(name.c_str());
    }
    /**
     * @brief Constructor for NVTX range with std::wstring name
     * @param name Range name as std::wstring
     */
    NvtxTracer(const std::wstring& name) {
        nvtxRangePushW(name.c_str());
    }
    /**
     * @brief Destructor - automatically pops the NVTX range
     */
    ~NvtxTracer() {
        nvtxRangePop();
    }
};

/** @brief NVTX range marker for profiling (when NVTX enabled) */
#define NVTX_RANGE(name) NvtxTracer LocalNvtxTracer(name);
#else
/** @brief Empty NVTX range marker (when NVTX disabled) */
#define NVTX_RANGE(name)
#endif

// ------------------------------------------------------------------------------------------
// Uniform error handling
//
// cv::Exception is used (rather than a project exception type) because the code is built on
// OpenCV's cv::cuda::GpuMat / Stream, whose own errors already arrive as cv::Exception.
//
//   CUDA_CHECK(expr)     CUDA runtime call; throws cv::Exception (GpuApiCallError) on failure
//   cublasSafeCall(expr) cuBLAS call;       throws cv::Exception (GpuApiCallError) on failure
//   CUDA_WARN(expr)      CUDA runtime call; logs on failure and evaluates to false. Use where
//                        throwing is not allowed (destructors) or the API returns bool.
//   CHECK(expr)          legacy alias of CUDA_CHECK
//
// Messages have the form "<API>: <expr> failed: <error name> (<description>)" + file/line.
// ------------------------------------------------------------------------------------------

/** @brief "CUDA: <expr> failed: <error name> (<description>)", shared by CUDA_CHECK and CUDA_WARN */
inline std::string gemm_cuda_error_message(cudaError_t err, const char *expr)
{
    return std::string("CUDA: ") + expr + " failed: " + cudaGetErrorName(err) + " ("
           + cudaGetErrorString(err) + ")";
}

/** @brief Implementation of CUDA_CHECK: throws cv::Exception (GpuApiCallError) at file:line in func
 *  unless err == cudaSuccess */
inline void gemm_cuda_check(cudaError_t err,
                            const char *expr,
                            const char *file,
                            const int line,
                            const char *func)
{
    if (cudaSuccess != err) {
        cv::error(cv::Error::GpuApiCallError, gemm_cuda_error_message(err, expr), func, file, line);
    }
}

/** @brief Implementation of CUDA_WARN: logs the error to std::cerr; returns err == cudaSuccess */
inline bool gemm_cuda_warn(cudaError_t err, const char *expr, const char *file, const int line)
{
    if (cudaSuccess != err) {
        std::cerr << gemm_cuda_error_message(err, expr) << " at " << file << ":" << line
                  << std::endl;
        return false;
    }
    return true;
}

/** @brief Implementation of cublasSafeCall: throws cv::Exception (GpuApiCallError) unless
 *  err == CUBLAS_STATUS_SUCCESS; the message carries cuBLAS's status name and description */
inline void ___cublasSafeCall(cublasStatus_t err,
                              const char *expr,
                              const char *file,
                              const int line,
                              const char *func)
{
    if (CUBLAS_STATUS_SUCCESS != err) {
        cv::error(cv::Error::GpuApiCallError,
                  std::string("CUBLAS: ") + expr + " failed: " + cublasGetStatusName(err) + " ("
                      + cublasGetStatusString(err) + ")",
                  func,
                  file,
                  line);
    }
}

// #expr turns the call into the message text; CV_Func is OpenCV's portable __func__
#define CUDA_CHECK(expr) gemm_cuda_check((expr), #expr, __FILE__, __LINE__, CV_Func)
#define CUDA_WARN(expr) gemm_cuda_warn((expr), #expr, __FILE__, __LINE__)
#define cublasSafeCall(expr) ___cublasSafeCall((expr), #expr, __FILE__, __LINE__, CV_Func)
#define CHECK(expr) CUDA_CHECK(expr)

/**
 * @brief Convert bytes to gigabytes
 * @param mem Memory size in bytes
 * @return Memory size in GB (binary units, 1 GB = 1024^3 bytes, as a double)
 */
inline auto GB(int64_t mem)
{
    return ((mem / 1024.0) / 1024) / 1024;
}

/**
 * @brief Convert bytes to megabytes
 * @param mem Memory size in bytes
 * @return Memory size in MB
 */
inline auto MB(int64_t mem)
{
    return ((mem / 1024.0) / 1024);
}

/**
 * @brief Convert bytes to kilobytes
 * @param mem Memory size in bytes
 * @return Memory size in KB
 */
inline auto KB(int64_t mem)
{
    return (mem / 1024.0);
}

/**
 * @struct mem_info_t
 * @brief Memory information structure with size and units
 */
struct mem_info_t
{
    /**
     * @brief Constructor for memory info
     * @param size Memory size value
     * @param units Unit string (KB, MB, GB)
     */
    mem_info_t(double size, const std::string &units)
        : size_(size)
        , units_(units)
    {}
    const double size_; ///< Memory size value
    const std::string units_; ///< Unit string (KB, MB, GB)
};

/**
 * @brief Output stream operator for mem_info_t
 * @param os Output stream
 * @param info Memory info structure
 * @return Reference to output stream
 */
inline std::ostream &operator<<(std::ostream &os, const mem_info_t &info)
{
    os << info.size_ << info.units_;
    return os;
}

/**
 * @brief Convert bytes to human-readable format
 * @param mem Memory size in bytes
 * @return mem_info_t with appropriate units (KB, MB, GB): below 1 MB in KB (so tiny sizes print as
 *         fractional KB), below 1 GB in MB, otherwise GB; binary units. Used by the device report.
 */
inline auto UBytes(int64_t mem)
{
    constexpr auto mb = 1024 * 1024;
    if (mem < mb) {
        return mem_info_t(KB(mem), "KB");
    }
    constexpr auto gb = mb * 1024;
    if (mem < gb) {
        return mem_info_t(MB(mem), "MB");
    }
    return mem_info_t(GB(mem), "GB");
}
