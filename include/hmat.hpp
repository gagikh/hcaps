/**
 * @file hmat.hpp
 * @brief Host matrices in pinned (page-locked) or managed memory, backed by a reuse pool
 * @author GEMM Project
 * @date 2025
 * 
 * "H" stands for host, not hierarchical. Contents:
 *  - MemoryPool: a cv::MatAllocator that gets memory from a CUDA allocation function
 *    (cudaHostAlloc or cudaMallocManaged) and keeps freed blocks for reuse instead of
 *    returning them to CUDA, because those allocations are slow;
 *  - HMat: a cv::Mat whose buffer is pinned host memory. Used for every host-side test matrix
 *    so that GpuMat::upload / download on a stream are truly asynchronous and fast (a pageable
 *    buffer forces a staged, synchronous copy);
 *  - MMat: the same with managed (unified) memory, plus a GpuMat view (currently unused);
 *  - PinnedAllocator / pinned_vector: a std::vector allocator on top of the pool (currently
 *    unused).
 *
 * Pitfalls:
 *  - pooled memory is only returned to CUDA by Release() (HMat::Release() is called at the end
 *    of the tests) or at static destruction; Release() while matrices are still alive leaves
 *    them dangling;
 *  - HMat derives from cv::Mat: assigning a plain cv::Mat makes the HMat share that matrix's
 *    buffer and allocator (cv::Mat::operator= copies both), so it is no longer pinned.
 *
 * The out-of-line members of MemoryPool and the two static pools are defined in hmat.cpp.
 */

#pragma once

#include <map>
#include <queue>
#include <mutex>
#include <unordered_map>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <QDebug>
#include <core.hpp>

namespace cv
{

/**
 * @class MemoryPool
 * @brief cv::MatAllocator that allocates with a CUDA function and recycles freed blocks
 * @tparam ALLOC_FCN CUDA allocation function with the signature (void**, size_t, unsigned flags),
 *                   e.g. cudaHostAlloc or cudaMallocManaged
 * @tparam DEALLOC_FCN Matching free function (cudaFreeHost or cudaFree)
 * @tparam TAG Flags passed as the third argument of ALLOC_FCN (cudaHostAllocDefault,
 *             cudaMemAttachGlobal); it also makes the two pool types distinct
 * 
 * Bookkeeping:
 *  - m_allocations: every block ever obtained from ALLOC_FCN and its size in bytes (whether
 *    in use or pooled); Release() frees exactly these;
 *  - m_pool: free blocks, grouped by their exact size. allocate() takes the smallest pooled
 *    block that is large enough, provided no more than m_allowedLostBytesRate (80 %) of it
 *    would be wasted, i.e. the block is less than 5x the request; otherwise it allocates anew.
 *  - deallocate() never frees: the block goes back to m_pool under its original size.
 * All three methods take m_mutex, so the pool can be shared between threads (see the note in
 * deallocate() about one unguarded lookup).
 */
template<cudaError_t (*ALLOC_FCN)(void** pHost, size_t size, unsigned int flags),
         cudaError_t(*DEALLOC_FCN)(void* pHost), int TAG>
class MemoryPool : public cv::MatAllocator
{
    // TAG: the allocation flags, e.g. cudaHostAllocDefault for pinned host memory
public:
    /**
     * @brief Default constructor for memory pool
     */
    MemoryPool() = default;

    /// Frees all blocks (see Release()); runs at static destruction for the global pools.
    ~MemoryPool()
    {
        Release();
    }

    /**
     * @brief Return every block ever allocated by this pool to CUDA and forget them
     * @warning Also frees blocks still owned by live matrices; call it only when no matrix
     *          using this pool is alive any more (e.g. at the end of main()).
     */
    void Release()
    {
	    const std::lock_guard<std::mutex> lock(m_mutex);
	    for (const auto& it : m_allocations)
	    {
		    const auto err = DEALLOC_FCN(it.first);
		    if (err != cudaSuccess) {
			    qCritical() << "Unable to free managed/host memory:" << gemm_cuda_error_message(err, "dealloc").c_str();
		    }
	    }
	    m_allocations.clear();
	    m_pool.clear();
    }

    /**
     * @brief cv::MatAllocator hook for UMat; a no-op here (this allocator only serves cv::Mat)
     * @param data UMatData pointer
     * @param accessflags Access flags
     * @param usageFlags Usage flags
     * @return Always true
     */
    bool allocate(cv::UMatData* data, cv::AccessFlag accessflags, cv::UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        (void)data;
        (void)accessflags;
        (void)usageFlags;
        return true;
    }

    /**
     * @brief cv::MatAllocator hook called by cv::Mat::create(): allocate (or reuse) a buffer
     * @param dims Number of dimensions
     * @param sizes Array of dimension sizes
     * @param type OpenCV element type
     * @param data0 User-supplied buffer, or nullptr to allocate; a user buffer is wrapped,
     *              marked USER_ALLOCATED and never pooled
     * @param step In / out byte steps per dimension; filled for a dense layout unless given
     *             together with @p data0
     * @return New UMatData describing the buffer (size = total bytes)
     * @throws cv::Exception if the CUDA allocation fails (CUDA_CHECK)
     */
    cv::UMatData* allocate(
        int dims,
        const int* sizes,
        int type,
        void* data0,
        size_t* step,
        cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE;

    /**
     * @brief cv::MatAllocator hook called when the last reference to a buffer goes away;
     *        returns the buffer to the pool (not to CUDA) and deletes @p u
     * @param u UMatData to deallocate (nullptr is ignored)
     */
    void deallocate(cv::UMatData* u) const CV_OVERRIDE;

    /**
     * @brief Allocate @p mat from this pool (sets mat.allocator, then mat.create())
     * @param mat Matrix to create
     * @param rows Number of rows
     * @param cols Number of columns
     * @param type Matrix type
     */
    inline void create(cv::Mat& mat, int rows, int cols, int type) {
        mat.allocator = this;
        mat.create(rows, cols, type);
    }

    /**
     * @brief Attach this pool as the allocator of @p mat without allocating; a later
     *        mat.create() (e.g. by GpuMat::download) then allocates from the pool
     * @param mat Matrix to attach
     */
    inline void create(cv::Mat& mat) {
        mat.allocator = this;
    }

private:
    // mutable: the cv::MatAllocator interface declares allocate / deallocate as const
    mutable std::map<size_t, std::queue<uchar*>> m_pool;        ///< free blocks by exact size (ordered for lower_bound)
    mutable std::unordered_map<uchar*, size_t> m_allocations;   ///< every block from ALLOC_FCN -> its size

    mutable std::mutex m_mutex;                                 ///< guards m_pool and m_allocations
    /// a pooled block is reused only if less than this fraction of it would be unused
    static constexpr float m_allowedLostBytesRate{ 0.8f };
};

/** @brief Pool of pinned (page-locked) host memory: cudaHostAlloc / cudaFreeHost */
typedef MemoryPool<cudaHostAlloc, cudaFreeHost, cudaHostAllocDefault> HostMemoryPool;
/** @brief Pool of managed (unified) memory accessible from host and device: cudaMallocManaged / cudaFree */
typedef MemoryPool<cudaMallocManaged, cudaFree, cudaMemAttachGlobal> ManagedMemoryPool;

/**
 * @class PinnedAllocator
 * @brief std::vector allocator using pinned memory from a HostMemoryPool
 * @tparam T Element type
 * @note Each allocator object is its own pool (it derives from HostMemoryPool), and the pool
 *       frees its blocks when the allocator is destroyed. Currently not used in the project.
 */
template <typename T>
class PinnedAllocator : public HostMemoryPool
{
public:
    using value_type = T; ///< Value type
    using pointer = value_type*; ///< Pointer type
    using size_type = std::size_t; ///< Size type

    PinnedAllocator() noexcept = default;

    /**
     * @brief Copy constructor for different types
     * @tparam U Different allocator type
     */
    template <typename U>
    PinnedAllocator(PinnedAllocator<U> const&) noexcept {}

    /**
     * @brief Allocate memory for n elements
     * @details Goes through the cv::MatAllocator interface (a 1-D uchar "matrix" of
     *          n * sizeof(T) bytes) and keeps only the raw pointer. The size is passed as int,
     *          so a single allocation must stay below 2 GiB.
     * @param n Number of elements to allocate
     * @return Pointer to allocated memory
     */
    auto allocate(size_type n) -> value_type* {
        int sizes[] = { int(n * sizeof(T)) };
        auto umat = HostMemoryPool::allocate(
            1,
            sizes,
            cv::DataType<uchar>::type,
            nullptr,
            nullptr,
            cv::AccessFlag::ACCESS_RW,
            cv::UMatUsageFlags::USAGE_ALLOCATE_HOST_MEMORY);
        auto ptr = umat->origdata;
        delete umat;
        return ptr;
    }

    /**
     * @brief Deallocate memory
     * @details Wraps @p p in a temporary UMatData with refcount 0 so that the pool's
     *          deallocate() takes it back (and deletes the temporary).
     * @param p Pointer to memory to deallocate
     * @param n Number of elements (unused)
     */
    auto deallocate(pointer p, size_type n) -> void {
        if (p) {
            auto u = new cv::UMatData(nullptr);
            u->data = u->origdata = p;
            u->urefcount = u->refcount = 0;
            u->flags = cv::UMatData::MemoryFlag::HOST_COPY_OBSOLETE;
            HostMemoryPool::deallocate(u);
        }
    }
};

/**
 * @brief Equality operator for PinnedAllocator
 * @tparam T First allocator type
 * @tparam U Second allocator type
 * @return Always true (as if the allocators were stateless; note each one actually owns a pool)
 */
template <class T, class U>
auto operator==(PinnedAllocator<T> const&, PinnedAllocator<U> const&) -> bool {
    return true;
}

/**
 * @brief Inequality operator for PinnedAllocator
 * @tparam T First allocator type
 * @tparam U Second allocator type
 * @return Always false (see operator==)
 */
template <class T, class U>
auto operator!=(PinnedAllocator<T> const&, PinnedAllocator<U> const&) -> bool {
    return false;
}

/**
 * @brief Template alias for a std::vector backed by pinned host memory
 * @tparam T Element type
 */
template <typename T>
using pinned_vector = std::vector<T, PinnedAllocator<T>>;

/**
 * @struct HMat
 * @brief cv::Mat whose buffer comes from the shared pinned host memory pool
 * @details Used for all host-side matrices that are uploaded to or downloaded from the GPU
 *          (test operands, results). A default-constructed HMat is empty but already has the
 *          pool as allocator, so `cv::HMat h; gpu.download(h, stream);` downloads into pinned
 *          memory asynchronously. Copies share the buffer (cv::Mat reference counting).
 */
struct HMat : public cv::Mat {
    /**
     * @brief Empty matrix that will allocate from the pinned pool on create()
     */
    HMat() : cv::Mat() {
        m_pool.create(*this);
    }

    /**
     * @brief Constructor with dimensions and type
     * @param rows Number of rows
     * @param cols Number of columns
     * @param type Matrix type (e.g., CV_32F)
     */
    HMat(int rows, int cols, int type) {
        m_pool.create(*this, rows, cols, type);
    }
    
    /**
     * @brief Constructor with size and type
     * @param size Matrix size
     * @param type Matrix type (e.g., CV_32F)
     */
    HMat(const cv::Size& size, int type) : HMat(size.height, size.width, type) {}

    /**
     * @brief Assignment operator: shares the buffer of @p h (no copy)
     * @param h Input matrix
     * @return Reference to this matrix
     * @warning If @p h was not allocated from the pool, this HMat now refers to non-pinned memory.
     */
    HMat& operator = (const cv::Mat& h) {
        cv::Mat::operator=(h);
        return *this;
    }

    /**
     * @brief Free all pinned memory of the shared pool (see MemoryPool::Release());
     *        called at the end of the test programs, when no HMat is alive any more
     */
    static void Release() { m_pool.Release(); }

    //explicit operator const cv::Mat& () const { return *this; }
    //__host__ explicit operator cv::Mat& () { return *this; }

private:
    static HostMemoryPool m_pool; ///< Pinned pool shared by all HMat objects (defined in hmat.cpp)
};

/**
 * @struct MMat
 * @brief cv::Mat whose buffer is managed (unified) memory, visible to host and device
 * @details gpu() gives a GpuMat view of the same memory, so no explicit copies are needed.
 *          Currently not used by the tests or the library.
 */
struct MMat : public cv::Mat {
    /**
     * @brief Default constructor
     */
    MMat() : cv::Mat() {
        m_pool.create(*this);
    }

    /**
     * @brief Constructor with dimensions and type
     * @param rows Number of rows
     * @param cols Number of columns
     * @param type Matrix type (e.g., CV_32F)
     */
    MMat(int rows, int cols, int type) {
        m_pool.create(*this, rows, cols, type);
    }
    
    /**
     * @brief Constructor with size and type
     * @param size Matrix size
     * @param type Matrix type (e.g., CV_32F)
     */
    MMat(const cv::Size& size, int type) : MMat(size.height, size.width, type) {}

    /**
     * @brief Assignment operator
     * @param h Input matrix
     * @return Reference to this matrix
     */
    MMat& operator = (const cv::Mat& h) {
        cv::Mat::operator=(h);
        return *this;
    }

    /**
     * @brief Get GPU matrix view of managed memory
     * @return GpuMat header over the same buffer (no copy, no ownership); valid while this
     *         MMat keeps the buffer alive. Host access must not overlap running kernels.
     */
    cv::cuda::GpuMat gpu() {
        return cv::cuda::GpuMat(this->rows, this->cols, this->type(), this->data, this->step);
    }

    /**
     * @brief Free all managed memory of the shared pool (see MemoryPool::Release())
     */
    static void Release() {
	    m_pool.Release();
    }

    //explicit operator const cv::Mat& () const { return *this; }
    //__host__ explicit operator cv::Mat& () { return *this; }

private:
    static ManagedMemoryPool m_pool; ///< Managed pool shared by all MMat objects (defined in hmat.cpp)
};

} // namespace
