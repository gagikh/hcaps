/**
 * @file hmat.cpp
 * @brief Out-of-line members of the pooled CUDA host allocator (MemoryPool) of hmat.hpp
 * @author GEMM Project
 * @date 2025
 * 
 * Defines MemoryPool::allocate() / deallocate() and the two static pools HMat::m_pool
 * (pinned host memory) and MMat::m_pool (managed memory). Defining those pools here is what
 * instantiates the two MemoryPool specializations, so other translation units only need the
 * declarations from the header.
 */

#include "hmat.hpp"
#include <QDebug>

namespace cv
{

/**
 * Allocate (or reuse) a buffer for a cv::Mat of the given dims / sizes / type.
 *
 * 1. Compute the byte size and, if requested, the dense steps (same algorithm as OpenCV's
 *    default allocator: innermost dimension first).
 * 2. A user-supplied buffer (data0) is only wrapped and flagged USER_ALLOCATED.
 * 3. Otherwise take the smallest pooled block >= size if it wastes less than 80 % of itself,
 *    else allocate a new block with ALLOC_FCN and record it in m_allocations.
 */
template<cudaError_t(*ALLOC_FCN)(void** pHost, size_t size, unsigned int flags),
         cudaError_t(*DEALLOC_FCN)(void* pHost), int TAG>
cv::UMatData* MemoryPool<ALLOC_FCN, DEALLOC_FCN, TAG>::allocate(
    int dims,
    const int* sizes,
    int type,
    void* data0,
    size_t* step,
    cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const
{
    // total = bytes of one element, grown dimension by dimension from the innermost one;
    // after the loop it is the size of the whole buffer
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        if (step)
        {
            // user buffer with explicit step: honour it (it may include row padding)
            if (data0 && step[i] != cv::Mat::AUTO_STEP) {
                CV_Assert(total <= step[i]);
                total = step[i];
            } else {
                step[i] = total;
            }
        }

        total *= sizes[i];
    }

    auto u = new cv::UMatData(this);
    u->size = total;

    // this is not our data: wrap it, never pool or free it
    if (data0) {
        u->data = u->origdata = static_cast<uchar*>(data0);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    } else {
        void* ptr = nullptr;
        {
            const std::lock_guard<std::mutex> lock(m_mutex);
            // smallest free block size >= total; reuse it only if the unused fraction
            // 1 - total / block_size is below m_allowedLostBytesRate (block < 5x the request)
            auto it = m_pool.lower_bound(total);
            if (m_pool.end() != it && (1.f - (float)total / it->first) < m_allowedLostBytesRate)
            {
                // invariant: empty queues are erased, so every key has at least one block
                CV_Assert(!it->second.empty());
                ptr = it->second.front();
                it->second.pop();
                if (it->second.empty()) {
                    m_pool.erase(it);
                }
            }
        }
        if (nullptr == ptr) {
            // pool miss: allocate a new block of exactly `total` bytes (TAG = allocation flags).
            // size_t, not int: int(total) overflowed for buffers >= 2 GiB (e.g. 32768 x 16384 floats)
            // and a failed allocation used to be logged and then used as nullptr; CUDA_CHECK throws
            CUDA_CHECK(ALLOC_FCN(&ptr, total, TAG));
            const std::lock_guard<std::mutex> lock(m_mutex);
            m_allocations[reinterpret_cast<uchar*>(ptr)] = total;
        }
        u->data = u->origdata = static_cast<uchar*>(ptr);
    }
    return u;
}

/**
 * Called by OpenCV when the last cv::Mat referencing @p u is released (refcount 0).
 * A block allocated by this pool is pushed back to m_pool under its original size (it is
 * NOT freed; Release() does that). User-allocated buffers are left alone. @p u is deleted.
 */
template<cudaError_t(*ALLOC_FCN)(void** pHost, size_t size, unsigned int flags),
         cudaError_t(*DEALLOC_FCN)(void* pHost), int TAG>
void MemoryPool<ALLOC_FCN, DEALLOC_FCN, TAG>::deallocate(cv::UMatData* u) const
{
    if (!u) {
        return;
    }

    CV_Assert(0 <= u->urefcount);
    CV_Assert(0 <= u->refcount);

    if (0 == u->refcount)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            // NOTE: this lookup runs before the lock is taken; it is safe only as long as no
            // other thread allocates from or releases the same pool concurrently
            auto it = m_allocations.find(u->origdata);
            if (it != m_allocations.end()) // our allocation
            {
                const std::lock_guard<std::mutex> lock(m_mutex);
                m_pool[it->second].push(it->first);
            } else {
                // Not a block of this pool (e.g. one handed out before Release(), which has
                // already been freed, or memory of another pool instance): do nothing.
                // This case was found by a crash dump and its origin is not fully understood.
                // The disabled delete would be wrong anyway: the memory never came from new.
                // delete u->origdata;
            }
            u->origdata = nullptr;
        }
        delete u;
    }
}

// The pools shared by all HMat / MMat objects. Being statics, they are destroyed at program exit
// (freeing all blocks); call HMat::Release() earlier to free while the CUDA context is still alive.
HostMemoryPool HMat::m_pool;
ManagedMemoryPool MMat::m_pool;
}
