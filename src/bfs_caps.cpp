/**
 * @file bfs_caps.cpp
 * @brief Host part of class BFSCaps (include/bfs_caps.hpp): tree layout, memory checks, leaf GEMM calls
 * @author GEMM Project
 * @date 2025
 * 
 * Contents, in order:
 *  - config_t::initialize(): L2-based heuristic for the row loop counts of the passes;
 *  - find_roi(), wrap_roi(), sums_rows(), create_intermediate_sums_layout(): packing of the S
 *    areas of all internal nodes into one buffer m_S;
 *  - required_memory(), reserved_memory(), fits_device_memory(): will (N, R) fit into the GPU;
 *  - initialize() (internal and public): builds the recursion tree (S areas, product buffers,
 *    address caches), the pointer arrays of the batched leaf GEMM, and
 *    uploads the node addresses to constant memory;
 *  - batched_cublas(), batched_cublas_ex(), batched_leaf(): the
 *    leaf level;
 *  - get_result() and the download_*() test helpers;
 * The device side (bfs_kernel, process<op>(), store_address_cache(), the tuner) is in src/bfs_caps.cu.
 * All buffers hold halves typed as CV_32FC1 (one float = two halves): a width or pitch in floats
 * is half the width in halves, hence the many "/ 2" and "N / 2" below.
 */

#include "bfs_caps.hpp"
#include <tuning.hpp>
#include <core.hpp>
#include <common.hpp> // create_continuous()

#include <opencv2/core/cuda.hpp>
#include <cuda_fp16.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <sstream>
#include <driver_types.h>
#include <cuda_runtime_api.h>

/// base^exponent in integers (exponent >= 0); replaces std::pow, whose double result was used
/// where an exact integer count is meant
static size_t integer_power(size_t base, int exponent)
{
    size_t result = 1;
    for (auto i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

/**
 * @brief Heuristic row loop counts of the passes from the L2 size of device 0
 *
 * For each pass an estimate of the bytes it touches is halved until it fits into L2 + 8%; the
 * number of halvings h gives loops = 2^h (each loop then works on 1 / loops of the rows).
 * Only BILINEAR and the reconstruction get a heuristic (R > 1); the ENCODE / DECODE branches run
 * for R < 1 only, which never happens (initialize() handles R = 0 without a config), so their
 * counts stay 1. initialize() overrides all four with gen/tuning_table.hpp when it has an entry.
 * The *_grid_size values are computed but not used.
 *
 * @note The device properties are queried once (function-local statics, device 0 only)
 * @see cudaDeviceProp for GPU property information
 */
void BFSCaps::config_t::initialize(int N, int R)
{
    static cudaDeviceProp prop;
    static bool initialized = false;
    if (!initialized) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        initialized = true;
    }
    const auto &L2_size = prop.l2CacheSize;
    const auto &SM = prop.multiProcessorCount;

    // a working set up to 8% above the L2 size still counts as fitting
    constexpr auto percent = 8;
    const auto boundary = (L2_size * (100.0 + percent) / 100.0);

    // half2 pairs per thread vector: 4 with half8 (16 bytes), 1 with half2
    constexpr auto T = sizeof(half8) / sizeof(half2);
    if (R < 1)
    {
        // transform A
        // for each iteration we
        // - run (N/2)x(N/(2 * T) threads
        // - read 3 * (N/2) * (N/2) halves
        // - write 1 * (N/2) * (N/2) halves
        const auto write_bytes = 1 * (N / 2) * (N / 2) * 2;
        const auto read_bytes = 3 * write_bytes;
        const auto memory_size = std::max(write_bytes, read_bytes);

        // take into account B also
        auto tmp = 2 * memory_size;
        auto stride = 1;
        while (tmp > boundary) {
            tmp /= 2;
            stride *= 2;
        }

        // calculate memory size for A and B transform
        m_transform_A_and_B_loops_count = stride;

        // now let's calculate the required threads per iteration (result unused)
        const auto warps_count_per_row = N / 2 / T / WARP_SIZE;
        const auto total_warps_per_iteration = (N / 2 / stride) * warps_count_per_row;
        const auto warps_per_SM = cv::divUp(total_warps_per_iteration, SM);
        const auto blocks_per_SM = cv::divUp(warps_per_SM, WARPS_IN_X_DIR * WARPS_IN_Y_DIR);
        const auto transform_A_and_B_grid_size = blocks_per_SM * SM;
    }

    if (R < 1)
    {
        // transform C
        // for each iteration we
        // - run (N/2)x(N/(2 * T) threads
        // - read 3 * (N/2) * (N/2) halves
        // - write 2 * (N/2) * (N/2) halves
        const auto write_bytes = 2 * (N / 2) * (N / 2) * 2;
        const auto read_bytes = 3 * write_bytes / 2;
        const auto memory_size = std::max(write_bytes, read_bytes);

        auto tmp = memory_size;
        auto stride = 1;
        while (tmp > boundary) {
            tmp /= 2;
            stride *= 2;
        }

        // the loop count of the C (decode) transform
        m_transform_C_loops_count = stride;

        // now let's calculate the required threads per iteration (result unused)
        const auto warps_count_per_row = N / 2 / T / WARP_SIZE;
        const auto total_warps_per_iteration = (N / 2 / stride) * warps_count_per_row;
        const auto warps_per_SM = cv::divUp(total_warps_per_iteration, SM);
        const auto blocks_per_SM = cv::divUp(warps_per_SM, WARPS_IN_X_DIR * WARPS_IN_Y_DIR);
        const auto transform_C_grid_size = blocks_per_SM * SM;
    }

    if (R > 1)
    {
        // bilinear
        // for each iteration l
        // - read 2 full rows of left and right mult matrix with size N
        // - write 6 rows of destination sums with size N/2
        // - read 2 * 2 * N
        // - write 3 * N
        // total memory for first iteration
        // 4N + 3N = 7N
        // next we will read same data with N/2 rows but produce  7 * 6 * N / 4
        // we don't need to observe the last layer
        // estimate in bytes: 2 N^2 halves (A and B) * 2 bytes, scaled by (7/4)^(R-1), the growth
        // of the data per level (7x the nodes, each a quarter of the size) down to level R - 1.
        // Exact in integers: N^2 (7/4)^(R-1) = 7^(R-1) (N >> (R-1))^2 (N = k * 2^R)
        const auto top = size_t(N >> (R - 1));
        const auto memory_size = size_t(2 * 2) * integer_power(MULTS_COUNT, R - 1) * top * top;

        auto tmp = memory_size;
        auto loops = 1;
        while (tmp > boundary) {
            tmp /= 2;
            loops *= 2;
        }
        // the loop count of the bilinear (sums) pass
        m_bilinear_loops_count = loops;
    }
    {
        // each bilinear loop covers N / loops rows of the top level
        const auto &loops = m_bilinear_loops_count;
        m_bilinear_rows_per_loop = N / loops;
    }

    if (R > 1) {
        // without first layer
        // bytes: 2 N^2 halves * 2 bytes * ((7/4)^R - 1), where (7/4)^R - 1 = 3/4 * the sum of
        // (7/4)^l over l = 0 .. R - 1 (the per-level growth summed over the levels).
        // Exact in integers: N^2 ((7/4)^R - 1) = (7^R - 4^R) (N >> R)^2
        const auto leaf = size_t(N >> R);
        const auto memory_size =
            size_t(2 * 2) * (integer_power(MULTS_COUNT, R) - integer_power(4, R)) * leaf * leaf;

        auto tmp = memory_size;
        auto loops = 1;
        while (tmp > boundary) {
            tmp /= 2;
            loops *= 2;
        }
        m_reconstruct_loops_count = loops;
    }
    {
        // the reconstruction loops run over the rows of the leaf blocks (N >> R)
        const auto last_layer = R;
        const auto M = N >> last_layer;
        const auto &loops = m_reconstruct_loops_count;
        m_reconstruct_rows_per_loop = M / loops;
    }
}

// Takes a height x width rectangle from the free rectangles left_rois (first fit, front first) and
// returns it; an empty rectangle when the front one is narrower or too short (the caller then
// grows the front rectangle downwards, see sums_rows()).
//  - front exactly as wide: the rectangle is cut from its top, the rest stays at the front;
//  - front wider: it is split into R1 (returned), R2 (right of R1, same height) and R3 (below
//    both, full width); R2 goes to the front so that the next narrower area fills it first.
// The list order matters: sums_rows() and initialize() must place the areas identically.
static auto find_roi(int height, int width, std::list<cv::Rect> &left_rois)
{
    CV_Assert(!left_rois.empty());
    auto &front = left_rois.front();
    cv::Rect roi;
    if (front.width == width) {
        roi = front;
        roi.height = height;
        CV_Assert(front.height >= height);
        front.height -= height;
        front.y += height;
        if (0 == front.height) {
            // fully used: remove it
            left_rois.pop_front();
        }
        return roi;
    } else {
        CV_Assert(width < front.width);
        if (front.height < height) {
            return cv::Rect();
        }
        // decompose roi into small pieces
        // |R1 R2|
        // |R3   |
        const cv::Rect R1(front.x, front.y, width, height);
        const cv::Rect R2(front.x + width, front.y, front.width - width, height);
        const cv::Rect R3(front.x, front.y + height, front.width, front.height - height);
        left_rois.pop_front();
        if (0 != R3.height) {
            left_rois.emplace_front(R3); // insertion order is important, do not change
        }
        if (0 != R2.height) {
            left_rois.emplace_front(R2);
        }
        return R1;
    }
}

/**
 * @brief Create GpuMat view from raw pointer with ROI (a rectangle of m_S given by find_roi())
 * @param ptr Raw pointer to matrix data (start of m_S)
 * @param roi Region of interest rectangle
 * @param lds Leading dimension (stride)
 * @return GpuMat view of the specified ROI
 */
inline auto wrap_roi(float *ptr, const cv::Rect &roi, int lds)
{
    // size_t: at N = 16384, R = 4 the S buffer is > 2^31 floats and roi.y * lds overflows int
    return cv::cuda::GpuMat(roi.height, roi.width, CV_32FC1, ptr + size_t(roi.y) * lds + roi.x, lds * sizeof(float));
}

// Rows of m_S (N / 2 floats each): replays the find_roi() packing of initialize() without
// allocating. Level l has 7^l nodes of M = N >> l halves; each needs an area of M halves (cols =
// M / 2 floats) x 3 cols rows = the six (M/2) x (M/2) sums S1 .. S6:
//     R = 1: 6 (N/2)^2,  R = 2: + 7 * 6 (N/4)^2,  R = 3: + 49 * 6 (N/8)^2, ...
// When no free rectangle is tall enough, the front one (always at x = 0, full width) is extended
// downwards by the missing rows. See BFSCaps::sums_rows() in bfs_caps.hpp for the layout and the numbers.
int BFSCaps::sums_rows(int N, int R)
{
    // because of halves
    const auto global_lds = N / 2;

    auto shrinked_rows = 0;
    std::list<cv::Rect> left_rois;

    auto count = 1;
    for (auto l = 0; l < R; ++l) {
        const auto M = N >> l;
        const auto cols = M / 2;

        for (auto i = 0; i < count; ++i) {
            if (left_rois.empty()) {
                shrinked_rows += 3 * cols;
                left_rois.emplace_back(0, 0, int(global_lds), int(3 * cols));
            }
            const auto roi = find_roi(3 * cols, cols, left_rois);
            if (roi.empty()) {
                auto &front = left_rois.front();
                CV_Assert(0 == front.x);
                const auto height = left_rois.front().height;
                const auto delta = 3 * cols - height;
                front.height += delta;
                shrinked_rows += delta;
            }
        }
        count *= MULTS_COUNT;
    }
    return shrinked_rows;
}

// One allocation for all S areas; initialize() then cuts the areas out of it with find_roi().
void BFSCaps::create_intermediate_sums_layout(int N, int R)
{
    m_S.create(sums_rows(N, R), N / 2, CV_32FC1);
}

// Mirrors the allocations of initialize(N, R); see bfs_caps.hpp for each term and examples.
size_t BFSCaps::required_memory(int N, int R)
{
    const auto n = size_t(N);
    const auto matrix = n * n * sizeof(half); // one N x N half matrix
    if (R <= 0) {
        return 3 * matrix; // A, B, m_C (plain GEMM)
    }
    size_t count = 1; // leaf products: 7^R
    for (auto r = 0; r < R; ++r) {
        count *= MULTS_COUNT;
    }
    const auto M = n >> R; // leaf size
    // the driver allocates large blocks in 2 MB pages: a 7 x leaf^2 allocation is rounded up
    constexpr size_t granularity = size_t(2) << 20;
    // one create_continuous() per node of level R - 1: its 7 products of M x M halves
    const auto per_node = (MULTS_COUNT * M * M * sizeof(half) + granularity - 1) / granularity * granularity;
    const auto sums = size_t(sums_rows(N, R)) * (n / 2) * sizeof(float); // m_S, rows of N / 2 floats
    const auto products = count / MULTS_COUNT * per_node;                // 7^(R-1) nodes
    const auto pointers = 3 * count * sizeof(half *);                    // AA, BB, CC of the batched GEMM
    // A and B (2 * matrix) belong to the caller; counted so the check works before allocating them
    return 2 * matrix + sums + products + pointers;
}

// A fixed margin of 2 GiB, the same as the batched benchmark's (tst/gemm_benchmark.cpp). It covers
// what required_memory() does not count: the CUDA context (a few hundred MB), the cuBLAS workspace
// (MBs to tens of MB per handle), CUDA graphs; the 2 MB
// rounding of the product buffers is already in required_memory(). The first rule, 5% of the
// device but at least 512 MB, kept 4 GB free on the 80 GB H100 and skipped N = 65536, R = 2. It
// does not cover memory of other processes (desktop, browser, other CUDA applications):
// fits_device_memory() compares against the total, not the free memory.
size_t BFSCaps::reserved_memory(size_t total)
{
    (void) total; // the margin no longer depends on the device size (kept in the interface)
    return size_t(2) << 30;
}

// "1234 MB" (MiB, no decimals) for the messages of fits_device_memory()
static std::string mbytes(size_t bytes)
{
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(0);
    ss << bytes / double(1 << 20) << " MB";
    return ss.str();
}

// required + reserved <= total. The total, not the free memory, on purpose: the same (N, R) are
// skipped on every run whatever else is allocated, and the check gives the same answer before and
// after the caller allocates A and B (initialize() repeats it). Memory of other processes is not
// seen; initialize() then fails on the allocation and releases everything. See bfs_caps.hpp.
bool BFSCaps::fits_device_memory(int N, int R, std::string *why)
{
    size_t free_bytes = 0, total_bytes = 0; // free_bytes is not used, see above
    if (cudaSuccess != cudaMemGetInfo(&free_bytes, &total_bytes)) {
        cudaGetLastError(); // clear
        if (why) {
            *why = "cudaMemGetInfo failed (no CUDA device?)";
        }
        return false;
    }
    const auto required = required_memory(N, R);
    const auto reserved = reserved_memory(total_bytes);
    if (required + reserved <= total_bytes) {
        return true;
    }
    if (why) {
        *why = "N=" + std::to_string(N) + ", R=" + std::to_string(R) + " needs " + mbytes(required) + " + "
               + mbytes(reserved) + " reserved, the GPU has " + mbytes(total_bytes);
    }
    return false;
}

// Internal part of initialize(): builds the recursion tree on the host side, R >= 1.
//  - allocates m_S and cuts the S area of every internal node out of it, level by level and
//    node by node in node order, with find_roi() (the packing sums_rows() replays);
//  - for each node of level R - 1 allocates one continuous buffer for its 7 leaf products
//    (7 x leaf rows of leaf halves, leaf = N >> R);
//  - records each node's operands and S area in m_intermediate_address_cache[l] (the table that
//    the public initialize() uploads to constant memory) and the product buffers in
//    m_mult_address_cache[R - 1];
//  - sets m_config: heuristic, then gen/tuning_table.hpp.
// Allocation failures throw (cv::Exception); the public initialize() catches them.
bool BFSCaps::initialize(const cv::cuda::GpuMat &A,
                     const cv::cuda::GpuMat &B,
                     int R)
{
    const auto &N = A.rows;
    CV_Assert(R >= 0 && ((N >> R) << R) == N); // N = k * 2^R
    // half data in a float matrix: N rows of N / 2 floats
    const auto use_half_precision = N == (2 * A.cols);
    CV_Assert(use_half_precision);
    if (0 >= R) {
        std::cerr << "Usual AxB should be called here" << std::endl;
        return false;
    }
    CV_Assert(CV_32FC1 == A.type());
    CV_Assert(A.size() == B.size());
    CV_Assert(A.type() == B.type());

    // make sure matrices are continuous, so strides are valid: the kernels assume a pitch of N
    // halves for A, B and m_S alike
    {
        const auto lda = int(A.step / sizeof(half));
        CV_Assert(lda == int(B.step / sizeof(half)));
        CV_Assert(lda == N);
    }
    m_intermediate_address_cache.clear();
    m_mult_address_cache.clear();
    {
        m_intermediate_sums.clear();
        m_leaf_multiplications.clear();

        m_intermediate_sums.resize(R);
        m_leaf_multiplications.resize(R);

        m_intermediate_address_cache.resize(R);
        m_mult_address_cache.resize(R);
    }
    create_intermediate_sums_layout(N, R);

    // the common row pitch in floats (N halves)
    const auto global_lds = N / 2;
    // free rectangles of m_S, initially all of it; find_roi() takes the areas from the front
    std::list<cv::Rect> left_rois;

    left_rois.emplace_back(0, 0, m_S.cols, m_S.rows);
    auto ptr = reinterpret_cast<float *>(m_S.data);

    // layer 0: the root, operands A and B, S area 3N/2 rows x N halves
    {
        constexpr auto l = 0;
        const auto M = N >> l;
        const auto cols = M / 2;

        cv::cuda::GpuMat intermediate_sum, leaf_mult;
        // we will use intermediates for backward pass in order to keep reconstructed mult results
        {
            // each Si has (N/2)x(N/2) halves, i.e. (N/2) rows x (N/4) floats
            // S1 S2
            // S3 S4
            // S5 S6
            // Si keeps half precision data
            // area: 3 * cols rows (3N/2) x cols floats (N halves, the full width of m_S)
            const auto roi = find_roi(3 * cols, cols, left_rois);
            CV_Assert(!roi.empty());
            intermediate_sum = wrap_roi(ptr, roi, global_lds);
        }

        if (R <= 1) { // we need to create OO in the last block to store leaf mults
            // R = 1: the root is the leaf parent; 7 products of (N/2)x(N/2) halves, stacked
            // vertically: 7 * N/2 rows x N/4 floats
            // not cv::cuda::createContinuous(): int overflow from 2^31 elements (N = 65536), see common.hpp
            create_continuous(MULTS_COUNT * cols, cols / 2, CV_32FC1, leaf_mult);
            m_mult_address_cache[l].emplace_back(leaf_mult.ptr<float>());
        }
        m_intermediate_address_cache[l].emplace_back(A, B, intermediate_sum);

        m_intermediate_sums[l].push_back(intermediate_sum);
        m_leaf_multiplications[l].push_back(leaf_mult);
    }

    // other layers: level l is built from the nodes of level l - 1; each parent of size M
    // (halves) gets its 7 children m1 .. m7 of size H = M / 2, in that order (node order)
    for (auto l = 1; l < R; ++l) {
        const auto M = N >> (l - 1);
        for (auto &parents : m_intermediate_address_cache[l - 1]) {
            auto &A = parents.m_L;
            auto &B = parents.m_R;

            auto A11 = block(A, M, 0, 0, global_lds);
            auto A12 = block(A, M, 0, 1, global_lds);
            auto A21 = block(A, M, 1, 0, global_lds);
            auto A22 = block(A, M, 1, 1, global_lds);

            auto B11 = block(B, M, 0, 0, global_lds);
            auto B12 = block(B, M, 0, 1, global_lds);
            auto B21 = block(B, M, 1, 0, global_lds);
            auto B22 = block(B, M, 1, 1, global_lds);

            // H: rows of a quadrant (= its size in halves), W: its width in floats
            const auto H = M / 2;
            const auto W = H / 2;

            // the parent's S area (3H rows x 2W floats) and its six H x H sums
            // we will use SS for backward pass in order to keep reconstructed mult results
            auto S = cv::cuda::GpuMat(3 * H, 2 * W, CV_32FC1, parents.m_S, global_lds * sizeof(float));
            auto S1 = S.rowRange(0 * H, 1 * H).colRange(0, 1 * W);
            auto S2 = S.rowRange(0 * H, 1 * H).colRange(W, 2 * W);
            auto S3 = S.rowRange(1 * H, 2 * H).colRange(0, 1 * W);
            auto S4 = S.rowRange(1 * H, 2 * H).colRange(W, 2 * W);
            auto S5 = S.rowRange(2 * H, 3 * H).colRange(0, 1 * W);
            auto S6 = S.rowRange(2 * H, 3 * H).colRange(W, 2 * W);

            const std::vector<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> tmp = {
                {A11, B11}, // m1 = A11 * B11
                {A12, B21}, // m2 = A12 * B21
                {A21, S4}, // m3 = A21 * S4
                {A22, B22}, // m4 = A22 * B22
                {S1, S5}, // m5 = S1 * S5
                {S2, S6}, // m6 = S2 * S6
                {S3, B12}, // m7 = S3 * B12
            };
            for (auto& entry : tmp) {
                // the child is H x H halves; its S area holds six (H/2)x(H/2) sums:
                // 3 * HH rows x 2 * WW floats (= H halves)
                const auto HH = H / 2;
                const auto WW = HH / 2;

                const auto roi = find_roi(3 * HH, 2 * WW, left_rois);
                CV_Assert(!roi.empty());
                auto intermediate_sum = wrap_roi(ptr, roi, global_lds);
                cv::cuda::GpuMat leaf_mult;

                if (l == (R - 1)) {
                    // a leaf parent: its 7 products of HH x HH halves (HH = N >> R), stacked
                    // vertically in one continuous buffer (pitch HH halves)
                    create_continuous(MULTS_COUNT * HH, HH / 2, CV_32FC1, leaf_mult); // see common.hpp
                    m_mult_address_cache[l].emplace_back(leaf_mult.ptr<float>());
                }
                m_intermediate_address_cache[l].emplace_back(entry.first, entry.second, intermediate_sum);
                m_intermediate_sums[l].push_back(intermediate_sum);
                m_leaf_multiplications[l].push_back(leaf_mult);
            }
        }
    }

    m_config.initialize(N, R);
    // tuned loop counts of this GPU (or the defaults) from gen/tuning_table.hpp; they are what the
    // default build compiles into the kernels, the heuristic above covers sizes without an entry
    tuning::bfs_loops_t loops;
    if (tuning::bfs_loops(tuning::device_name(), N, R, loops)) {
        m_config.m_transform_A_and_B_loops_count = loops.encode;
        m_config.m_bilinear_loops_count = loops.bilinear;
        m_config.m_reconstruct_loops_count = loops.reconstruct;
        m_config.m_transform_C_loops_count = loops.decode;
        m_config.m_bilinear_rows_per_loop = N / loops.bilinear;
        m_config.m_reconstruct_rows_per_loop = (N >> R) / loops.reconstruct;
    }
    return true;
}

// The pointer arrays of the batched leaf GEMM: for each node j of level R - 1 (node order) its 7
// products m1 .. m7 = (A11, B11), (A12, B21), (A21, S4), (A22, B22), (S1, S5), (S2, S6), (S3, B12)
// (quadrants of the node's operands and the sums in its S area, see src/bfs_caps.cu), written to the
// product buffer of node j, product i at rows i * M (M = N >> R). Entry 7 * j + i.
std::vector<BFSCaps::leaf_product_t> BFSCaps::leaf_products(int N, int R) const
{
    const auto global_lds = N / 2; // A, B and S: pitch N / 2 floats = N halves
    const auto l = R - 1;
    // M: leaf size in halves; the nodes of level R - 1 are 2M x 2M
    const auto M = N >> (l + 1);
    const auto H = M;
    const auto W = H / 2; // M halves = M / 2 floats

    std::vector<leaf_product_t> products;
    products.reserve(m_intermediate_address_cache[l].size() * MULTS_COUNT);
    auto buffer = m_mult_address_cache[l].begin(); // product buffer of node j (lists have no operator[])
    for (const auto &node : m_intermediate_address_cache[l]) {
        const auto A11 = block(node.m_L, M << 1, 0, 0, global_lds);
        const auto A12 = block(node.m_L, M << 1, 0, 1, global_lds);
        const auto A21 = block(node.m_L, M << 1, 1, 0, global_lds);
        const auto A22 = block(node.m_L, M << 1, 1, 1, global_lds);
        const auto B11 = block(node.m_R, M << 1, 0, 0, global_lds);
        const auto B12 = block(node.m_R, M << 1, 0, 1, global_lds);
        const auto B21 = block(node.m_R, M << 1, 1, 0, global_lds);
        const auto B22 = block(node.m_R, M << 1, 1, 1, global_lds);

        // the node's S area: six M x M sums S1 S2 / S3 S4 / S5 S6
        const cv::cuda::GpuMat S(3 * H, 2 * W, CV_32FC1, node.m_S, global_lds * sizeof(float));
        const auto S1 = S.rowRange(0 * H, 1 * H).colRange(0, 1 * W);
        const auto S2 = S.rowRange(0 * H, 1 * H).colRange(W, 2 * W);
        const auto S3 = S.rowRange(1 * H, 2 * H).colRange(0, 1 * W);
        const auto S4 = S.rowRange(1 * H, 2 * H).colRange(W, 2 * W);
        const auto S5 = S.rowRange(2 * H, 3 * H).colRange(0, 1 * W);
        const auto S6 = S.rowRange(2 * H, 3 * H).colRange(W, 2 * W);

        // left and right operands of m1 .. m7 (same pairs as initialize())
        const cv::cuda::GpuMat left[] = {A11, A12, A21, A22, S1, S2, S3};
        const cv::cuda::GpuMat right[] = {B11, B21, S4, B22, S5, S6, B12};
        const cv::cuda::GpuMat mults(MULTS_COUNT * H, W, CV_32FC1, *buffer++); // dense: pitch M halves
        for (auto i = 0; i < MULTS_COUNT; ++i) {
            products.push_back({left[i], right[i], mults.rowRange(i * H, (i + 1) * H)});
        }
    }
    return products;
}

// m_mult_buffers from leaf_products()
void BFSCaps::prepare_mult_buffers(int N, int R)
{
    thrust::host_vector<half *> AA;
    thrust::host_vector<half *> BB;
    thrust::host_vector<half *> CC;
    const auto global_ldm = N >> R; // the product buffers are dense: pitch M halves
    for (const auto &p : leaf_products(N, R)) {
        AA.push_back(reinterpret_cast<half *>(p.L.data));
        BB.push_back(reinterpret_cast<half *>(p.R.data));
        CC.push_back(reinterpret_cast<half *>(p.C.data));
        CV_Assert(size_t(global_ldm) == p.C.step / sizeof(half));
    }
    // host -> device copies of the pointer arrays
    m_mult_buffers.AA = AA;
    m_mult_buffers.BB = BB;
    m_mult_buffers.CC = CC;
    // operands live in A, B or m_S: pitch N halves
    m_mult_buffers.ld_ab = N;
    m_mult_buffers.ld_c = global_ldm;
}

// Public initialize(): resets the object, validates N and R, checks the memory budget, then
// (R >= 1) builds the tree (internal initialize()) and the leaf GEMM pointer arrays, and uploads the node tables to constant memory on @p stream.
// R = 0 only allocates m_C. Every failure prints a message and returns false.
bool BFSCaps::initialize(const cv::cuda::GpuMat &A,
                     const cv::cuda::GpuMat &B,
                     int R,
                     cv::cuda::Stream &stream)
{
    m_intermediate_sums.clear();
    m_intermediate_address_cache.clear();

    m_leaf_multiplications.clear();
    m_mult_address_cache.clear();
    m_mult_buffers = {};
    m_S.release(); // a re-initialize frees the previous buffers before allocating new ones
    m_C.release();
    m_N = 0;
    m_R = 0;

    const auto& N = A.rows;

    if (R < 0 || R > MAX_RECURSIONS) {
        std::cerr << "Number of recursions R=" << R << " is out of range 0.." << MAX_RECURSIONS << std::endl;
        return false;
    }
    if (((N >> R) << R) != N) {
        std::cerr << "N=" << N << " is not k * 2^R for R=" << R << std::endl;
        return false;
    }
    // just keep headers
    m_A = A;
    m_B = B;
    const auto use_half_precision = N == (2 * A.cols);
    CV_Assert(use_half_precision);
    m_N = N;
    m_R = R;

    // everything below allocates: check that (N, R) fits into this GPU first, and treat an
    // allocation that fails anyway (fragmentation, other processes) the same way
    const auto release = [this] {
        m_intermediate_sums.clear();
        m_intermediate_address_cache.clear();
        m_leaf_multiplications.clear();
        m_mult_address_cache.clear();
        m_mult_buffers = {};
        m_S.release();
        m_C.release();
        m_N = m_R = 0;
    };
    {
        std::string why;
        if (!fits_device_memory(N, R, &why)) {
            std::cerr << "BFSCaps::initialize: " << why << std::endl;
            release();
            return false;
        }
    }
    try {
        if (0 == R) {
            // no recursion: batched_cublas() runs one plain GEMM into m_C, the other stages do nothing
            m_C.create(A.size(), A.type());
            return true;
        }

        if (!initialize(A, B, R)) {
            release();
            return false;
        }
        prepare_mult_buffers(N, R);
        if (!prepare_leaf_plan(m_mult_buffers)) { // wgmma: tensor maps of all 7^R products
            release();
            return false;
        }
    } catch (const cv::Exception &e) {
        std::cerr << "BFSCaps::initialize: N=" << N << ", R=" << R << ": " << e.err << std::endl;
        release();
        return false;
    } catch (const std::exception &e) { // thrust::system::bad_alloc, std::bad_alloc
        std::cerr << "BFSCaps::initialize: N=" << N << ", R=" << R << ": " << e.what() << std::endl;
        release();
        return false;
    }

    // flatten the per-level lists into the constant memory tables, in node order
    // TODO: use pinned memory here
    std::vector<layer_info_t> intermediate_cache;
    std::vector<float*> mult_cache;

    // TODO: calculate and set correct value (only a capacity hint, push_back grows as needed)
    // 1 + 7 + 49 + ...
    intermediate_cache.reserve(MULTS_COUNT);
    mult_cache.reserve(MULTS_COUNT);

    for (auto l = 0; l < R; ++l) {
        for (auto it = m_intermediate_address_cache[l].begin(); it != m_intermediate_address_cache[l].end(); ++it) {
            intermediate_cache.push_back(*it);
        }
    }
    {
        // only the last level (R - 1) has product buffers
        auto &container = m_mult_address_cache.back();
        for (auto it = container.begin(); it != container.end(); ++it) {
            mult_cache.push_back(*it);
        }

    }
    // note: on this failure m_S, the product buffers and m_N / m_R stay as they are (freed by the
    // next initialize() or the destructor)
    if (!store_address_cache(intermediate_cache, mult_cache, stream)) {
        std::cerr << "Critial, unable to store address cahce" << std::endl;
        m_intermediate_sums.clear();
        m_intermediate_address_cache.clear();
        m_mult_address_cache.clear();
        return false;
    }
    return true;
}

BFSCaps::~BFSCaps()
{
    // release memory (the member destructors would do it anyway)
    m_intermediate_sums.clear();
}

// Leaf GEMM with cuBLAS (cublasHgemmBatched, half accumulation): one call over the 7^R entries
// of m_mult_buffers, C[e] = A[e] * B[e] of leaf x leaf halves. R = 0: m_C = A * B.
void BFSCaps::batched_cublas(cv::cuda::Stream &stream)
{
    const auto N = get_dimension();
    const auto R = get_recursions();
    const auto leaf_dimension = N >> R;
    m_cgemm.set_stream(stream);
    if (0 == R) {
        m_cgemm.gemm(m_A, m_B, m_C); // no recursion: plain GEMM
        return;
    }
    m_cgemm.gemm(m_mult_buffers.AA,
                 m_mult_buffers.ld_ab,
                 m_mult_buffers.BB,
                 m_mult_buffers.ld_ab,
                 m_mult_buffers.CC,
                 m_mult_buffers.ld_c,
                 leaf_dimension);
}

// As batched_cublas() with an explicit cuBLAS algorithm and compute type (cublasGemmBatchedEx).
// R = 0 runs the plain gemm(), ignoring algo and compute.
void BFSCaps::batched_cublas_ex(cv::cuda::Stream &stream,
                            cublasGemmAlgo_t algo,
                            cublasComputeType_t compute)
{
    const auto N = get_dimension();
    const auto R = get_recursions();
    const auto leaf_dimension = N >> R;
    m_cgemm.set_stream(stream);
    if (0 == R) {
        m_cgemm.gemm(m_A, m_B, m_C); // no recursion: plain GEMM
        return;
    }
    m_cgemm.gemm_ex(m_mult_buffers.AA,
                    m_mult_buffers.ld_ab,
                    m_mult_buffers.BB,
                    m_mult_buffers.ld_ab,
                    m_mult_buffers.CC,
                    m_mult_buffers.ld_c,
                    leaf_dimension,
                    algo,
                    compute);
}

// The leaf GEMM of the pipeline: compile-time choice (CMake LEAF_GEMM -> LEAF_GEMM_WGMMA, see
// core.hpp) between our wgmma kernel and cuBLAS with the compute type matching @p accumulator.
void BFSCaps::batched_leaf(cv::cuda::Stream &stream, wgmma_gemm_t::accumulator_t accumulator)
{
    if (0 == get_recursions()) { // no recursion: one plain GEMM (LEAF_GEMM=wgmma: cuBLAS)
        batched_cublas_ex(stream, CUBLAS_GEMM_DFALT_TENSOR_OP,
                          accumulator == wgmma_gemm_t::accumulator_t::half ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F);
        return;
    }
    leaf_gemm(stream, m_mult_buffers, accumulator);
}

// See bfs_caps.hpp.
void BFSCaps::leaf_gemm(cv::cuda::Stream &stream, mult_buffers_t &buffers, wgmma_gemm_t::accumulator_t accumulator)
{
    CV_Assert(get_recursions() >= 1);
    const auto leaf = get_dimension() >> get_recursions();
#if (LEAF_GEMM_WGMMA)
    // one fixed tile configuration, nothing tuned yet; the plan holds the tensor maps of exactly
    // these operands (gemm() checks that it was made from buffers.AA)
    CV_Assert(buffers.wgmma_plan.valid() && buffers.wgmma_plan.dimension == leaf);
    m_wgmma_gemm.set_stream(stream);
    m_wgmma_gemm.gemm(buffers.wgmma_plan, buffers.AA, buffers.CC, buffers.ld_c, accumulator);
#else
    m_cgemm.set_stream(stream);
    m_cgemm.gemm_ex(buffers.AA, buffers.ld_ab, buffers.BB, buffers.ld_ab, buffers.CC, buffers.ld_c, leaf,
                    CUBLAS_GEMM_DFALT_TENSOR_OP,
                    accumulator == wgmma_gemm_t::accumulator_t::half ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F);
#endif
}

// See bfs_caps.hpp. Only LEAF_GEMM=wgmma has per-batch state: one TMA tensor map per operand.
bool BFSCaps::prepare_leaf_plan(mult_buffers_t &buffers)
{
    buffers.wgmma_plan = {};
#if (LEAF_GEMM_WGMMA)
    const auto leaf = get_dimension() >> get_recursions();
    std::string why;
    if (!wgmma_gemm_t::available(why)) {
        std::cerr << "BFS: LEAF_GEMM=wgmma cannot run here: " << why << std::endl;
        return false;
    }
    if (!wgmma_gemm_t::supports(leaf)) {
        std::cerr << "BFS: LEAF_GEMM=wgmma needs a leaf size that is a multiple of " << wgmma_gemm_t::BN
                  << ", got " << leaf << " (N = " << get_dimension() << ", R = " << get_recursions() << ")"
                  << std::endl;
        return false;
    }
    buffers.wgmma_plan = wgmma_gemm_t::make_plan(buffers.AA, buffers.ld_ab, buffers.BB, buffers.ld_ab, leaf);
#endif
    return true;
}

void BFSCaps::download_intermediate_sums(std::map<int, std::vector<std::vector<cv::Mat>>>& HS,
                                     cv::cuda::Stream &stream)
{
    HS.clear();
    for (auto l = 0; l < (int)m_intermediate_sums.size(); ++l) {
        for (auto& groups : m_intermediate_sums[l]) {
            // the six sums of the node's S area (3H x 2W floats)
            auto S1 = block(groups, 0, 0);
            auto S2 = block(groups, 0, 1);
            auto S3 = block(groups, 1, 0);
            auto S4 = block(groups, 1, 1);
            auto S5 = block(groups, 2, 0);
            auto S6 = block(groups, 2, 1);
            std::vector<cv::Mat> dst(ARITHEMTIC_OPS_COUNT);
            std::vector<cv::cuda::GpuMat> src = { S1, S2, S3, S4, S5, S6 };
            for (auto j = 0; j < ARITHEMTIC_OPS_COUNT; ++j) {
                src[j].download(dst[j], stream);
            }
            HS[l].push_back(dst);
        }
    }
    stream.waitForCompletion();
}

cv::cuda::GpuMat BFSCaps::get_result()
{
    if (0 == m_R) {
        return m_C; // plain GEMM
    }
    // the root S area (3N/2 rows x N/2 floats): after the backward pass its top 2H = N rows hold
    // the decoded product C (where S1 .. S4 were)
    constexpr auto l = 0;
    auto groups = m_intermediate_sums[l].front();
    {
        const auto H = groups.rows / 3;
        auto CC = groups.rowRange(0 * H, 2 * H);
        return CC;
    }
}

void BFSCaps::download_intermediate_mults(std::vector<std::vector<cv::Mat>> &HS,
                                      cv::cuda::Stream &stream)
{
    HS.clear();
    HS.resize(m_intermediate_sums.size());
    for (auto l = 0; l < (int) m_intermediate_sums.size(); ++l) {
        for (auto &groups : m_intermediate_sums[l]) {
            // the node's product: the top 2/3 of its S area (M x M halves)
            const auto H = groups.rows / 3;
            auto CC = groups.rowRange(0 * H, 2 * H);
            cv::Mat C;
            CC.download(C, stream);
            stream.waitForCompletion();
            HS[l].push_back(C);
        }
    }
    stream.waitForCompletion();
}

void BFSCaps::download_leaf_mults(std::vector<std::vector<cv::Mat>> &HM, cv::cuda::Stream &stream)
{
    HM.clear();
    for (auto l = 0; l < (int)m_leaf_multiplications.size(); ++l) {
        for (auto& groups : m_leaf_multiplications[l]) {
            // 7 products stacked vertically; empty buffers (nodes above level R - 1) give H = 0
            const auto H = groups.rows / MULTS_COUNT;
            std::vector<cv::Mat> dst(MULTS_COUNT);
            for (auto i = 0; i < MULTS_COUNT; ++i) {
                auto MM = groups.rowRange(i * H, (i + 1) * H);
                MM.download(dst[i], stream);
            }
            HM.push_back(dst);
        }
    }
    stream.waitForCompletion();
}

