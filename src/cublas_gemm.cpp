/**
 * @file cublas_gemm.cpp
 * @brief Implementation of CUBLAS GEMM wrapper class
 * @author GEMM Project
 * @date 2025
 * 
 * This file contains the implementation of the CUBLAS GEMM wrapper class
 * that provides a convenient interface for CUBLAS matrix multiplication operations.
 *
 * The project includes the legacy <cublas.h> (see core.hpp), so the handle-based API is called
 * by its explicit _v2 names (cublasCreate_v2, cublasSgemm_v2, ...).
 *
 * Every GEMM here uses the row-major trick described in cublas_gemm.hpp: cuBLAS computes the
 * column-major C^T = B^T * A^T, therefore B is passed first and A second, and the m / n sizes
 * are swapped accordingly.
 */

#include <cublas_gemm.hpp>
#include <core.hpp>

#include <opencv2/core/cuda.hpp>

/**
 * @brief Constructor for CUBLAS GEMM wrapper
 * 
 * Initializes CUBLAS handle with optimal settings for GEMM operations,
 * including TF32 tensor operations for improved performance.
 */
cublas_gemm_t::cublas_gemm_t()
{
    cublasSafeCall(cublasCreate_v2(&m_handle));
    cublasSafeCall(cublasSetPointerMode_v2(m_handle, CUBLAS_POINTER_MODE_HOST));
    // TF32 tensor cores for the float (CV_32F square) path below; the half paths are not affected
    cublasSafeCall(cublasSetMathMode(m_handle, cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH));
}

// delegates to the default constructor, then binds the handle to stream
cublas_gemm_t::cublas_gemm_t(cv::cuda::Stream& stream) : cublas_gemm_t()
{
    set_stream(stream);
}

/**
 * @brief Destructor for CUBLAS GEMM wrapper
 * 
 * Properly destroys the CUBLAS handle and cleans up resources.
 * The status is ignored on purpose: a destructor must not throw (cublasSafeCall would).
 */
cublas_gemm_t::~cublas_gemm_t()
{
    cublasDestroy_v2(m_handle);
}

void cublas_gemm_t::set_stream(cv::cuda::Stream &stream)
{
    // only call cuBLAS when the stream actually changes (callers set it before every GEMM)
    auto ptr = (cudaStream_t)stream.cudaPtr();
    if (ptr != m_stream) {
        cublasSafeCall(cublasSetStream_v2(m_handle, ptr));
        m_stream = ptr;
    }
}

void cublas_gemm_t::gemm(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::GpuMat &C)
{
    // cv::gemm()-style transposition flags, fixed to "no transposition"; the tr1 / tr2 code
    // paths below are kept for completeness but are never taken
    constexpr auto flags = 0;
    CV_Assert(A.type() == CV_32FC1);
    CV_Assert(B.type() == A.type());

    const auto tr1 = (flags & cv::GEMM_1_T) != 0;
    const auto tr2 = (flags & cv::GEMM_2_T) != 0;

    // swapped on purpose: cuBLAS's first operand is B (row-major trick), so B's flag goes to transa
    const auto transa = tr2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto transb = tr1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (A.rows == A.cols) {
        // float matrices (TF32 tensor cores via the handle's math mode)
        const auto ASize = tr1 ? cv::Size(A.rows, A.cols) : A.size();
        const auto BSize = tr2 ? cv::Size(B.rows, B.cols) : B.size();

        const cv::Size CSize(BSize.width, ASize.height);
        C.create(CSize, A.type());

        CV_Assert(ASize.width == BSize.height);
        CV_Assert(0 == A.step % sizeof(float));
        CV_Assert(0 == B.step % sizeof(float));
        CV_Assert(0 == C.step % sizeof(float));
        const auto alphaf = 1.0f;
        const auto bettaf = 0.0f;
        // m = columns of C (= of B), n = rows of C (= of A), k = inner dimension; ld = step in floats
        cublasSafeCall(cublasSgemm_v2(m_handle,
                                      transa,
                                      transb,
                                      tr2 ? B.rows : B.cols,
                                      tr1 ? A.cols : A.rows,
                                      tr2 ? B.cols : B.rows,
                                      &alphaf,
                                      B.ptr<float>(),
                                      static_cast<int>(B.step / sizeof(float)),
                                      A.ptr<float>(),
                                      static_cast<int>(A.step / sizeof(float)),
                                      &bettaf,
                                      C.ptr<float>(),
                                      static_cast<int>(C.step / sizeof(float))));
    } else {
        // half precision packed as CV_32FC1: an N x N half matrix is stored as N x N/2 floats.
        // Only square matrices are used in this project, so all cuBLAS sizes are N.
        const auto N = A.rows;
        CV_Assert(N == 2 * A.cols);
        CV_Assert(B.size() == A.size());
        C.create(A.size(), A.type());

        CV_Assert(0 == A.step % sizeof(half));
        CV_Assert(0 == B.step % sizeof(half));
        CV_Assert(0 == C.step % sizeof(half));

        // CUBLAS_POINTER_MODE_HOST: alpha / beta are read from host memory, as halves (the scale
        // type of a half GEMM); ld = row step in halves
        const auto alphaf = __float2half(1.0f);
        const auto bettaf = __float2half(0.0f);
        cublasSafeCall(cublasHgemm(m_handle,
                                   transa,
                                   transb,
                                   N, N, N,
                                   &alphaf,
                                   B.ptr<half>(),
                                   static_cast<int>(B.step / sizeof(half)),
                                   A.ptr<half>(),
                                   static_cast<int>(A.step / sizeof(half)),
                                   &bettaf,
                                   C.ptr<half>(),
                                   static_cast<int>(C.step / sizeof(half))));
    }
}

void cublas_gemm_t::gemm(const thrust::device_vector<half *> &A,
                         int lda,
                         const thrust::device_vector<half *> &B,
                         int ldb,
                         thrust::device_vector<half *> &C,
                         int ldc,
                         int dimension)
{
    // batch size = number of pointers; B and C are assumed to hold as many
    const auto count = A.size();
    const auto alphaf = __float2half(1.0f);
    const auto bettaf = __float2half(0.0f);
    // device addresses of the pointer arrays (cuBLAS reads the pointers on the GPU)
    auto A_start = thrust::raw_pointer_cast(&A[0]);
    auto B_start = thrust::raw_pointer_cast(&B[0]);
    auto C_start = thrust::raw_pointer_cast(&C[0]);
    // B before A: row-major trick (see the file comment)
    cublasSafeCall(cublasHgemmBatched(m_handle,
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      dimension,
                                      dimension,
                                      dimension,
                                      &alphaf,
                                      B_start,
                                      ldb,
                                      A_start,
                                      lda,
                                      &bettaf,
                                      C_start,
                                      ldc,
                                      count));
}

void cublas_gemm_t::gemm_ex(const thrust::device_vector<half *> &A,
                            int lda,
                            const thrust::device_vector<half *> &B,
                            int ldb,
                            thrust::device_vector<half *> &C,
                            int ldc,
                            int dimension,
                            cublasGemmAlgo_t algo,
                            cublasComputeType_t compute)
{
    const auto count = A.size();
    // alpha and beta have the compute type's scale type: half for CUBLAS_COMPUTE_16F, float for
    // every CUBLAS_COMPUTE_32F* (cuBLAS reads 4 bytes there: a half 1.0 read as float is ~0)
    const auto half_scale = compute == CUBLAS_COMPUTE_16F || compute == CUBLAS_COMPUTE_16F_PEDANTIC;
    const half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    const float alpha_f = 1.0f, beta_f = 0.0f;
    const void *alpha = half_scale ? static_cast<const void *>(&alpha_h) : static_cast<const void *>(&alpha_f);
    const void *beta = half_scale ? static_cast<const void *>(&beta_h) : static_cast<const void *>(&beta_f);
    auto A_start = thrust::raw_pointer_cast(&A[0]);
    auto B_start = thrust::raw_pointer_cast(&B[0]);
    auto C_start = thrust::raw_pointer_cast(&C[0]);

    // half inputs and outputs (CUDA_R_16F) in every case; only the accumulation follows `compute`.
    // B before A: row-major trick (see the file comment)
    cublasSafeCall(cublasGemmBatchedEx(m_handle,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       dimension,
                                       dimension,
                                       dimension,
                                       alpha,
                                       reinterpret_cast<void* const*>(B_start),
                                       cudaDataType::CUDA_R_16F,
                                       ldb,
                                       reinterpret_cast<void* const*>(A_start),
                                       cudaDataType::CUDA_R_16F,
                                       lda,
                                       beta,
                                       reinterpret_cast<void* const*>(C_start),
                                       cudaDataType::CUDA_R_16F,
                                       ldc,
                                       count,
                                       compute,
                                       algo));
}
