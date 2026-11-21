/**
 * @file cublas_gemm.hpp
 * @brief CUBLAS GEMM wrapper class for matrix multiplication operations
 * @author GEMM Project
 * @date 2025
 * 
 * This header contains the CUBLAS GEMM wrapper class that provides a convenient
 * interface for CUBLAS matrix multiplication operations with tensor core support.
 *
 * Where it is used:
 *  - BFSCaps::m_cgemm: the plain GEMM of R = 0 and the batched leaf GEMM of the CAPS pipeline when the
 *    leaves run on cuBLAS (BFSCaps::batched_cublas(), BFSCaps::batched_cublas_ex(), BFSCaps::leaf_gemm() with LEAF_GEMM=cublas);
 *  - the tests and benchmarks: the cuBLAS baseline CAPS is compared against.
 *
 * Row-major convention: the project stores matrices row-major, cuBLAS assumes column-major. A
 * row-major matrix read column-major is its transpose, so every call computes C^T = B^T * A^T,
 * i.e. passes B as cuBLAS's first operand and A as the second. The result, read row-major, is
 * C = A * B, and no explicit transposition is needed.
 *
 * Half matrices travel as CV_32FC1 GpuMats: an N x N half matrix is an N x N/2 float matrix (two
 * halves per float), see gemm(const GpuMat &, ...). All leading dimensions passed to the
 * pointer-array overloads are in half elements.
 */

#pragma once

#include <cublas.h>
#ifndef Q_MOC_RUN // Qt's moc cannot parse CUDA 13 Thrust (CCCL 3 namespace macros)
#include <thrust/device_vector.h>
#endif

#include <opencv2/core/cuda.hpp>

/**
 * @struct cublas_gemm_t
 * @brief CUBLAS GEMM wrapper for matrix multiplication operations
 * 
 * Provides a convenient interface for CUBLAS matrix multiplication operations
 * with automatic tensor core utilization and stream management.
 *
 * Owns one cuBLAS handle (non-copyable). Scalars are passed in host memory
 * (CUBLAS_POINTER_MODE_HOST) with alpha = 1, beta = 0: C is overwritten, never accumulated.
 * All operations are asynchronous on the stream of the last set_stream() (the legacy default
 * stream before the first call).
 */
struct cublas_gemm_t
{
    /**
     * @brief Constructor for CUBLAS GEMM wrapper: creates the handle, host pointer mode, TF32 math
     *        mode (affects the float path of gemm(GpuMat) only)
     * @throws cv::Exception if cuBLAS cannot be initialized (e.g. no CUDA device)
     */
    cublas_gemm_t();
    
    /**
     * @brief Constructor for CUBLAS GEMM wrapper with stream
     * @param stream CUDA stream for asynchronous operations
     */
    cublas_gemm_t(cv::cuda::Stream &stream);
    
    /**
     * @brief Destructor for CUBLAS GEMM wrapper
     */
    ~cublas_gemm_t();

    // owns a cuBLAS handle: copying would destroy it twice
    cublas_gemm_t(const cublas_gemm_t &) = delete;
    cublas_gemm_t &operator=(const cublas_gemm_t &) = delete;

    /**
     * @brief Set CUDA stream for operations
     * @param stream CUDA stream to use for operations
     * @details Calls cublasSetStream() only when the stream differs from the last one, so it is
     *          cheap to call before every operation (BFSCaps does).
     */
    void set_stream(cv::cuda::Stream &stream);
    
    /**
     * @brief Get CUBLAS handle
     * @return CUBLAS handle for direct operations
     */
    inline auto handle() const { return m_handle; }

    /**
     * @brief Perform GEMM operation using CUBLAS with OpenCV GpuMat
     * @param A First input matrix (GpuMat)
     * @param B Second input matrix (GpuMat)
     * @param C Output matrix (GpuMat)
     * 
     * Wrapper implementation for CUBLAS GEMM utilizing tensor cores.
     * Handle is configured to use tensors. Computes C = A * B.
     *
     * Two cases, distinguished by the shape of A (both CV_32FC1):
     *  - A square: a float GEMM (cublasSgemm, TF32 tensor cores; the result is TF32-rounded);
     *  - A.rows == 2 * A.cols: N x N half matrices packed two per float, cublasHgemm (half
     *    accumulation), all sizes N.
     * B must have the same shape as A. C is (re)allocated with C.create() when its size or type
     * differ, so during CUDA graph capture C must already have the right shape (no allocation
     * may happen while capturing). The implementation has a transposition flag fixed to 0, so
     * transposed operands are not supported.
     */
    void gemm(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::GpuMat &C);
    
    /**
     * @brief Perform GEMM operation with half-precision data
     * @details Batched: C[i] = A[i] * B[i] for i < A.size() with cublasHgemmBatched (half inputs,
     *          half accumulation, cuBLAS's default algorithm). Every product is
     *          dimension x dimension, row-major.
     * @param A First input matrix: device array of pointers to the left operands
     * @param lda Leading dimension of A (in halves; the same for all products)
     * @param B Second input matrix: device array of pointers to the right operands
     * @param ldb Leading dimension of B (in halves)
     * @param C Output matrix: device array of pointers to the products (overwritten)
     * @param ldc Leading dimension of C (in halves)
     * @param dimension Dimension of the matrices
     * @pre A, B and C are non-empty and have the same size (not checked). The pointer arrays are
     *      read by the GPU when the kernel runs, so they must stay alive (and unchanged, if the call
     *      was captured into a CUDA graph) until then.
     */
    void gemm(const thrust::device_vector<half*> &A,
              int lda,
              const thrust::device_vector<half*> &B,
              int ldb,
              thrust::device_vector<half*> &C,
              int ldc,
              int dimension);
    /**
     * @brief Batched GEMM of half matrices with a chosen algorithm and compute type
     *        (cublasGemmBatchedEx); same operands, layout and preconditions as the batched gemm()
     * @param A First input matrix
     * @param lda Leading dimension of A
     * @param B Second input matrix
     * @param ldb Leading dimension of B
     * @param C Output matrix
     * @param ldc Leading dimension of C
     * @param dimension Dimension of the matrices
     * @param algo Algorithm to use (callers pass CUBLAS_GEMM_DFALT_TENSOR_OP)
     * @param compute Accumulation type. CUBLAS_COMPUTE_16F (default, the historical behaviour)
     *        accumulates in half precision; CUBLAS_COMPUTE_32F accumulates in float with half
     *        inputs/outputs. This decides which cuBLAS baseline CAPS is compared against.
     */
    void gemm_ex(const thrust::device_vector<half *> &A,
                 int lda,
                 const thrust::device_vector<half *> &B,
                 int ldb,
                 thrust::device_vector<half *> &C,
                 int ldc,
                 int dimension,
                 cublasGemmAlgo_t algo,
                 cublasComputeType_t compute);

private:
    cublasHandle_t m_handle = nullptr; ///< owned cuBLAS handle
    cudaStream_t m_stream = nullptr;   ///< stream last given to cublasSetStream() (nullptr: default stream)
};
