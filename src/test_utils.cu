/**
 * @file test_utils.cu
 * @brief Device helpers for the tests: random matrix generation and half <-> float conversion
 * @author GEMM Project
 * @date 2025
 * 
 * Implements the GPU functions declared in common.hpp:
 *  - urand(): fill a CV_32FC1 / CV_32SC1 GpuMat with uniform values in (-range, +range];
 *  - to_half(): N x N float GpuMat -> half-packed N x N/2 CV_32FC1 GpuMat;
 *  - to_float(): the inverse conversion.
 * All are built on two small generic kernels (for_each_tuple) launched with 32 x 4 blocks,
 * one thread per element. Nothing here is performance-critical; it only prepares test data.
 *
 * Half-packed layout: an N x N half matrix is stored in an OpenCV matrix of type CV_32FC1
 * with N/2 columns, i.e. each float slot holds two consecutive halves of a row. The kernels
 * address such a matrix with a typed pointer (GlobPtr<half>) and the row step in bytes,
 * which is the same for both views.
 */

#include <curand_kernel.h>
#include <cuda_fp16.h>

#include <opencv2/cudev/ptr2d/glob.hpp>

/**
 * @brief Generator kernel: dst(y, x) = D(op(x, y)) for every element
 * @tparam D Destination element type
 * @tparam Op Functor with a __device__ operator()(int x, int y)
 * @param dst Destination matrix (typed pointer, byte row step, rows / cols in elements of D)
 * @param op Generator called with the column x and row y of the element
 * 
 * 2D grid, one thread per element; threads outside the matrix exit.
 */
template<class D, class Op>
__global__ void for_each_tuple(cv::cudev::GlobPtrSz<D> dst, Op op)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows) {
        return;
    }
    dst.row(y)[x] = D(op(x, y));
}

/**
 * @brief Map kernel: dst(y, x) = D(op(src(y, x))) for every element of src
 * @tparam S Source element type
 * @tparam D Destination element type
 * @tparam Op Functor with a __device__ operator()(S)
 * @param src Source matrix; its rows / cols (in elements of S) define the iteration space
 * @param dst Destination matrix without size information; it must hold at least src.rows rows
 *            of src.cols elements of D, and is indexed with the same (y, x) as src
 * @param op Conversion applied to each element
 * 
 * Used for the half <-> float conversions, where src and dst have the same number of
 * elements per row but a different element size.
 */
template<class S, class D, class Op>
__global__ void for_each_tuple(cv::cudev::GlobPtrSz<S> src, cv::cudev::GlobPtr<D> dst, Op op)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.cols || y >= src.rows) {
        return;
    }
    const auto in = src.row(y)[x];
    dst.row(y)[x] = D(op(in));
}

/**
 * @brief One uniform random number in (-range, +range] for the element with index @p offset
 * @param seed Random seed (same for the whole matrix)
 * @param offset Linear element index, used as the curand subsequence number
 * @param range Half-width of the output interval
 * @return (2u - 1) * range with u = curand_uniform() in (0, 1], i.e. a value in (-range, range]
 * 
 * Each thread initializes its own XORWOW state with subsequence = element index, so values are
 * independent across elements and reproducible for a given seed. curand_init with a nonzero
 * subsequence is relatively expensive, which is acceptable for test data generation.
 */
__device__ auto generate_uniform_random(unsigned long long seed, long offset, int range)
{
    curandState state;
    curand_init(seed, offset, 0, &state); // Initialize state with unique index
    const auto val = curand_uniform(&state); // Generate uniform random number in (0,1]
    return (2 * val - 1) * range;
}

/// Generator functor for urand(): element (y, x) of a matrix with m_N columns gets the random
/// number of subsequence x + y * m_N (row-major linear index), scaled to [-m_range, m_range].
struct urand_t
{
    int m_N;
    int m_range;
    unsigned long long m_seed;
    urand_t(int N, int range, unsigned long long seed)
        : m_N(N)
        , m_range(range)
        , m_seed(seed)
    {}

    __device__ auto operator()(int x, int y)
    {
        return generate_uniform_random(m_seed, x + y * m_N, m_range);
    }
};

/**
 * @brief Launch the generator kernel on @p dst (asynchronous, on @p stream)
 * @tparam D Element type the kernel writes (float or int for urand)
 * @param dst Allocated destination; the data are written through its device pointer even
 *            though the reference is const (GpuMat is a shallow handle)
 * @note The grid is sized from dst.cols, while the kernel bound uses cols in elements of D;
 *       both agree because D has the matrix's element size (4 bytes) at all call sites.
 */
template<class D, class Op>
void for_each_element(const cv::cuda::GpuMat &dst, Op op, cv::cuda::Stream &stream)
{
    // 32 threads in x: one warp covers 32 consecutive elements of a row (coalesced)
    const dim3 block(32, 4);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    // row length in elements of D (row bytes / sizeof(D))
    const auto cols = dst.cols * dst.elemSize() / sizeof(D);
    for_each_tuple<<<grid, block, 0, (cudaStream_t) stream.cudaPtr()>>>(
        cv::cudev::globPtr<D>(reinterpret_cast<D *>(dst.data), dst.step, dst.rows, cols), op);
}

/**
 * @brief Launch the map kernel src -> dst (asynchronous, on @p stream)
 * @tparam S Element type read from src (float or half)
 * @tparam D Element type written to dst (half or float)
 * @param src Source matrix; the number of S elements per row is src row bytes / sizeof(S)
 * @param dst Destination, already allocated with that many D elements per row
 * @note Both matrices are viewed with the same element count per row; e.g. for to_half a row
 *       of N floats (src) maps to a row of N halves (dst, stored as N/2 float slots).
 */
template<class S, class D, class Op>
void for_each_element(const cv::cuda::GpuMat &src,
                      cv::cuda::GpuMat &dst,
                      Op op,
                      cv::cuda::Stream &stream)
{
    const dim3 block(32, 4);
    // row length in elements of S: for a half-packed source this is twice src.cols
    const auto cols = src.cols * src.elemSize() / sizeof(S);

    const dim3 grid(cv::cudev::divUp(cols, block.x), cv::cudev::divUp(src.rows, block.y));

    for_each_tuple<<<grid, block, 0, (cudaStream_t) stream.cudaPtr()>>>(
        cv::cudev::globPtr<S>(reinterpret_cast<S *>(src.data), src.step, src.rows, cols),
        cv::cudev::globPtr<D>(reinterpret_cast<D *>(dst.data), dst.step, dst.rows, cols),
        op);
}

/// Fill @p C (CV_32FC1 or CV_32SC1, already allocated) with uniform random values in
/// (-range, +range]; for CV_32SC1 the float value is truncated toward zero. Asynchronous.
void urand(int range, unsigned long long seed, cv::cuda::GpuMat &C, cv::cuda::Stream &stream)
{
    urand_t op(C.cols, range, seed);
    if (C.type() == CV_32FC1) {
        for_each_element<float, decltype(op)>(C, op, stream);
    } else {
        CV_Assert(CV_32SC1 == C.type());
        for_each_element<int, decltype(op)>(C, op, stream);
    }
}

/// float -> half, round to nearest even
struct to_half_t
{
    __device__ auto operator()(float f) { return __float2half(f); }
};

/// Square N x N CV_32FC1 -> half-packed N x N/2 CV_32FC1 (N halves per row). Asynchronous.
void to_half(const cv::cuda::GpuMat &A, cv::cuda::GpuMat &AA, cv::cuda::Stream &stream)
{
    CV_Assert(A.type() == CV_32FC1);
    CV_Assert(A.cols == A.rows);
    AA.create(A.rows, A.rows / 2, A.type()); // N/2 float slots = N halves per row

    to_half_t op;
    for_each_element<float, half, decltype(op)>(A, AA, op, stream);
}

/// half -> float (exact)
struct to_float_t
{
    __device__ auto operator()(half h) { return __half2float(h); }
};

/// Half-packed N x N/2 CV_32FC1 (N halves per row) -> N x N CV_32FC1. Asynchronous.
void to_float(const cv::cuda::GpuMat &A, cv::cuda::GpuMat &AA, cv::cuda::Stream &stream)
{
    CV_Assert(A.type() == CV_32FC1);
    CV_Assert(A.cols * 2 == A.rows); // half-packed square matrix: N rows, N/2 float slots
    AA.create(A.rows, A.rows, A.type());

    to_float_t op;
    for_each_element<half, float, decltype(op)>(A, AA, op, stream);
}

/// Convenience overload returning a new matrix; the result is valid once @p stream reaches it.
cv::cuda::GpuMat to_float(const cv::cuda::GpuMat& A, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat AA;
    to_float(A, AA, stream);
    return AA;
}

// L2 reuse probe (declared in common.hpp): the access pattern of the BFS passes without their
// arithmetic, for test_diagnostics l2-reuse
__global__ void l2_probe_write(float4 *p, size_t n)
{
    const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        __stcg(p + i, v);
    }
}

__global__ void l2_probe_read(const float4 *p, size_t n, int *sink)
{
    int acc = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float4 v = __ldcg(p + i);
        acc ^= __float_as_int(v.x) ^ __float_as_int(v.y) ^ __float_as_int(v.z) ^ __float_as_int(v.w);
    }
    // keep the loads: one int per block (the value itself is not used)
    for (int offset = 16; offset > 0; offset /= 2) {
        acc ^= __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (threadIdx.x == 0 && acc == 0x7fffffff) {
        sink[blockIdx.x] = acc;
    }
}

void l2_probe(bool write, void *buffer, size_t bytes, int *sink, int sms, cudaStream_t stream)
{
    const size_t n = bytes / sizeof(float4);
    const int blocks = 8 * sms;
    if (write) {
        l2_probe_write<<<blocks, 256, 0, stream>>>(static_cast<float4 *>(buffer), n);
    } else {
        l2_probe_read<<<blocks, 256, 0, stream>>>(static_cast<const float4 *>(buffer), n, sink);
    }
}

// Bulk store / reduce-add probe (declared in common.hpp): one 16 KB shared-memory tile per block,
// written chunk after chunk with the TMA bulk copy or bulk reduce-add (f16 add, no flush to zero)
constexpr int BULK_PROBE_CHUNK = 16384;

__global__ void bulk_probe_kernel(char *dst, size_t chunks, int reduce)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) unsigned char bulk_tile[];
    half *h = reinterpret_cast<half *>(bulk_tile);
    for (int i = threadIdx.x; i < BULK_PROBE_CHUNK / 2; i += blockDim.x) {
        h[i] = __float2half(1.0f / 1024.0f); // small: repeated adds stay finite
    }
    // generic-proxy writes of the tile, then the async (bulk copy) proxy reads it
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        const auto src = static_cast<unsigned>(__cvta_generic_to_shared(bulk_tile));
        for (size_t c = blockIdx.x; c < chunks; c += gridDim.x) {
            char *d = dst + c * BULK_PROBE_CHUNK;
            if (reduce) {
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
                             :
                             : "l"(d), "r"(src), "r"(BULK_PROBE_CHUNK)
                             : "memory");
            } else {
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                             :
                             : "l"(d), "r"(src), "r"(BULK_PROBE_CHUNK)
                             : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 7;" ::: "memory"); // at most 8 in flight
        }
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); // all writes done
    }
#endif
}

void bulk_probe(bool reduce, void *dst, size_t bytes, int sms, cudaStream_t stream)
{
    const size_t chunks = bytes / BULK_PROBE_CHUNK;
    bulk_probe_kernel<<<4 * sms, 128, BULK_PROBE_CHUNK, stream>>>(static_cast<char *>(dst), chunks, reduce ? 1 : 0);
}
