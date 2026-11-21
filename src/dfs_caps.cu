/**
 * @file dfs_caps.cu
 * @brief class DFSCaps (include/dfs_caps.hpp): depth-first top level, leaf products added into C
 * @author GEMM Project
 * @date 2026
 *
 * Device side: three small memory-bound kernels: one level of the alternative-basis transform over
 * a whole matrix (the same elementwise operations as src/alternative_basis.cuh, without the
 * cooperative launch and row loops of bfs_kernel), the formation of a sub-problem operand and the
 * six sums of every node of one sub-tree level (as bilinear.cuh). All on 8 halves per thread.
 */

#include "dfs_caps.hpp"
#include <common.hpp> // create_continuous(), get_SM_count()
#include <core.hpp> // half8
#include "utils.cuh"             // half8 operator+ / -, fetch_cache_L2(), store_cache_L2()
#include "alternative_basis.cuh" // encoder_functor_t, decoder_functor_t (the BFS kernels' arithmetic)
#include "bilinear.cuh"          // bilinear<>::sums() (the six sums of a node)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef Q_MOC_RUN
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif
#include <algorithm>
#include <type_traits>
#include <sstream>

namespace {

constexpr int THREADS = 256;

/**
 * Grid of the element-wise kernels: x covers the 8-half groups of one row (THREADS per block), z
 * the blocks or nodes the launch works on, y the rows (grid-stride), with about 8 blocks per SM in
 * all. A thread never divides by a size: N = k * 2^R need not be a power of two, only the number
 * of blocks per side of a level is (shift and mask).
 */
dim3 grid_for(size_t groups_per_row, size_t rows, size_t count)
{
    static const size_t budget = size_t(get_SM_count()) * 8; // common.hpp: queried once
    const size_t gx = cv::divUp(groups_per_row, size_t(THREADS));
    const size_t gy = std::max<size_t>(1, std::min(rows, budget / std::max<size_t>(1, gx * count)));
    return dim3(unsigned(gx), unsigned(gy), unsigned(count));
}

/**
 * One level of the alternative-basis transform on a matrix with leading dimension ld: every
 * 2M x 2M block of it is seen as quadrants a b / c d (M x M), and
 *   encode (A and B):  d <- b + d - c
 *   decode (A only):   b <- b - d,  c <- d - c
 * (encoder_functor_t / decoder_functor_t, alternative_basis.cuh). One thread per half8 (8 halves,
 * 16 B) of the d quadrants; blockIdx.z is the block (2^log_blocks of them per side, so its row and
 * column are a shift and a mask), y the row inside the quadrant (grid-stride), x the 8-half group.
 * M % 8 == 0, ld % 8 == 0; loads and stores keep the lines in L2 (the next level reads them).
 * @tparam DECODE false: encode, true: decode (resolved at compile time)
 */
template <bool DECODE>
__global__ void transform_level_kernel(half *__restrict__ A, half *__restrict__ B, unsigned int ld, unsigned int M,
                                       unsigned int log_blocks)
{
    const size_t x = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) << 3;
    if (x >= M) {
        return;
    }
    const size_t bi = blockIdx.z >> log_blocks;
    const size_t bj = blockIdx.z & ((1u << log_blocks) - 1);
    const size_t top = bi * 2 * M + M;          // first row of the d quadrant
    const size_t left = bj * 2 * M + M + x;     // this thread's column in it
    for (size_t r = blockIdx.y; r < M; r += gridDim.y) {
        const size_t d = (top + r) * ld + left;
        if constexpr (DECODE) {
            decoder_functor_t<true>::runLR(A + d, A + d, ld, M, false);
        } else {
            encoder_functor_t<true>::runLR(A + d, B + d, ld, M, false);
        }
    }
}

/// How form_operand_kernel combines its sources
enum class form_t { copy, add, subtract };

/// dst (m x m, dense: leading dimension m) = p (copy), p + q (add) or p - q (subtract); p, q:
/// leading dimension ld; m % 8 == 0, ld % 8 == 0. x: 8-half group of a row, y: row (grid-stride)
template <form_t FORM>
__global__ void form_operand_kernel(half *__restrict__ dst, const half *__restrict__ p, const half *__restrict__ q,
                                    unsigned int ld, unsigned int m)
{
    const size_t x = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) << 3;
    if (x >= m) {
        return;
    }
    for (size_t y = blockIdx.y; y < m; y += gridDim.y) {
        const size_t i = (y * ld + x) >> 3;
        half8 v = fetch_cache_L2(reinterpret_cast<const half8 *>(p) + i);
        if constexpr (FORM == form_t::add) {
            v = v + fetch_cache_L2(reinterpret_cast<const half8 *>(q) + i);
        } else if constexpr (FORM == form_t::subtract) {
            v = v - fetch_cache_L2(reinterpret_cast<const half8 *>(q) + i);
        }
        store_cache_L2(reinterpret_cast<half8 *>(dst) + ((y * m + x) >> 3), v);
    }
}

/// Node of the sub-tree as the sums kernel sees it (same layout as DFSCaps::node_t)
struct sum_node_t
{
    const half *L, *R;
    half *S;
};

/**
 * The six sums of every node of one sub-tree level (quadrant size h, every buffer with pitch ld
 * halves) into the node's S area laid out as S1 S2 / S3 S4 / S5 S6 (bilinear<>::sums(),
 * bilinear.cuh): S1 = A21 + A22, S2 = A22 - A12, S3 = A22 - A11 (A = L), S4 = B22 - B11,
 * S5 = B21 + B22, S6 = B22 - B12 (B = R). blockIdx.z: the node, y: row of the quadrant
 * (grid-stride), x: its 8-half group. h % 8 == 0.
 */
__global__ void sums_kernel(const sum_node_t *__restrict__ nodes, unsigned int ld, unsigned int h)
{
    const unsigned int x8 = blockIdx.x * blockDim.x + threadIdx.x; // in half8
    if ((x8 << 3) >= h) {
        return;
    }
    const sum_node_t n = nodes[blockIdx.z];
    for (unsigned int r = blockIdx.y; r < h; r += gridDim.y) {
        bilinear<true, true, true>::sums(reinterpret_cast<const half8 *>(n.L), reinterpret_cast<const half8 *>(n.R),
                                         reinterpret_cast<half8 *>(n.S), ld >> 3, h, x8, r, false);
    }
}

/// Decoding matrix W: per product m1 .. m7 its (quadrant row, quadrant column, sign) terms
struct term_t
{
    int r, c, s;
};
const std::vector<std::vector<term_t>> &decoding()
{
    static const std::vector<std::vector<term_t>> W = {{{0, 0, 1}},
                                                       {{0, 0, 1}, {1, 1, -1}},
                                                       {{1, 0, 1}},
                                                       {{1, 1, -1}},
                                                       {{0, 1, 1}, {1, 1, 1}},
                                                       {{1, 0, 1}, {1, 1, 1}},
                                                       {{0, 1, -1}}};
    return W;
}

/// Operand of root product i: quadrants (row, column) of the encoded matrix, combined as p + s q
struct operand_t
{
    int pr, pc;     // first quadrant
    int qr, qc;     // second quadrant, qr < 0: none
    int negative;   // 1: p - q
};
// m1 = A11 B11, m2 = A12 B21, m3 = A21 S4, m4 = A22 B22, m5 = S1 S5, m6 = S2 S6, m7 = S3 B12 with
// S1 = A21 + A22, S2 = A22 - A12, S3 = A22 - A11, S4 = B22 - B11, S5 = B21 + B22, S6 = B22 - B12
const operand_t LEFT[7] = {{0, 0, -1, 0, 0}, {0, 1, -1, 0, 0}, {1, 0, -1, 0, 0}, {1, 1, -1, 0, 0},
                           {1, 0, 1, 1, 0},  {1, 1, 0, 1, 1},  {1, 1, 0, 0, 1}};
const operand_t RIGHT[7] = {{0, 0, -1, 0, 0}, {1, 0, -1, 0, 0}, {1, 1, 0, 0, 1}, {1, 1, -1, 0, 0},
                            {1, 0, 1, 1, 0},  {1, 1, 0, 1, 1},  {0, 1, -1, 0, 0}};

/// Launches form_operand_kernel for operand o of the encoded matrix X (leading dimension N) into dst
void form_operand(half *dst, half *X, const operand_t &o, int N, cudaStream_t stream)
{
    const unsigned int H = unsigned(N) >> 1;
    const auto quadrant = [&](int r, int c) { return X + size_t(r) * H * size_t(N) + size_t(c) * H; };
    const half *p = quadrant(o.pr, o.pc);
    const dim3 grid = grid_for(H >> 3, H, 1);
    if (o.qr < 0) {
        form_operand_kernel<form_t::copy><<<grid, THREADS, 0, stream>>>(dst, p, nullptr, unsigned(N), H);
    } else if (o.negative) {
        form_operand_kernel<form_t::subtract><<<grid, THREADS, 0, stream>>>(dst, p, quadrant(o.qr, o.qc), unsigned(N), H);
    } else {
        form_operand_kernel<form_t::add><<<grid, THREADS, 0, stream>>>(dst, p, quadrant(o.qr, o.qc), unsigned(N), H);
    }
}

std::string mbytes(size_t bytes)
{
    return std::to_string(bytes >> 20) + " MB";
}

} // namespace

size_t DFSCaps::sums_rows(int H, int levels)
{
    // level l: 7^l nodes of size m = H >> l, 2^l side by side per band of 3m/2 rows
    size_t rows = 0, nodes = 1;
    for (int l = 0; l < levels; ++l, nodes *= 7) {
        const size_t per_band = size_t(1) << l;
        const size_t bands = cv::divUp(nodes, unsigned(per_band));
        rows += bands * ((3 * (size_t(H) >> l)) >> 1);
    }
    return rows;
}

size_t DFSCaps::required_memory(int N, int R, mode_t mode)
{
    const size_t n = size_t(N), H = n >> 1;
    const size_t per_set = 2 * H * H + sums_rows(int(H), R - 1) * H; // operands + sums, halves
    const size_t sets = mode == mode_t::overlapped ? 2 : 1;
    return (3 * n * n + sets * per_set) * sizeof(half);
}

bool DFSCaps::fits_device_memory(int N, int R, mode_t mode, std::string *why)
{
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSuccess != cudaMemGetInfo(&free_bytes, &total_bytes)) {
        cudaGetLastError();
        if (why) {
            *why = "cudaMemGetInfo failed";
        }
        return false;
    }
    const auto required = required_memory(N, R, mode);
    const size_t reserved = size_t(2) << 30; // as BFSCaps::reserved_memory()
    if (required + reserved <= total_bytes) {
        return true;
    }
    if (why) {
        *why = "DFSCaps N=" + std::to_string(N) + ", R=" + std::to_string(R) + " needs " + mbytes(required) + " + "
               + mbytes(reserved) + " reserved, the GPU has " + mbytes(total_bytes);
    }
    return false;
}

DFSCaps::~DFSCaps()
{
    release();
}

void DFSCaps::release()
{
    m_graph.reset();
    for (auto e : m_events) {
        cudaEventDestroy(e);
    }
    m_events.clear();
    if (m_form_stream != nullptr) {
        cudaStreamDestroy(m_form_stream);
        m_form_stream = nullptr;
    }
    m_sets.clear();
    m_targets.reset();
    m_C.release();
    m_N = m_R = 0;
}

// One buffer set: operands, sum buffer, the sub-tree's node tables (device) and the leaf plan.
// Node order breadth first, children m1 .. m7 of each node in order, as BFS: leaf 7 j + i is
// product m(i+1) of leaf parent j, which is the order of the target lists.
bool DFSCaps::build_set(set_t &set, std::string &why)
{
    const int H = m_N >> 1;
    const int levels = m_R - 1;
    create_continuous(H, H >> 1, CV_32FC1, set.subA);
    create_continuous(H, H >> 1, CV_32FC1, set.subB);
    create_continuous(int(sums_rows(H, levels)), H >> 1, CV_32FC1, set.S);
    half *S = reinterpret_cast<half *>(set.S.ptr<float>());
    std::vector<node_t> level = {{reinterpret_cast<half *>(set.subA.ptr<float>()), reinterpret_cast<half *>(set.subB.ptr<float>()), nullptr}};
    std::vector<half *> AA, BB;
    size_t band_row = 0; // first row of the current level's bands in S
    set.levels.clear();
    set.counts.clear();
    for (int l = 0; l < levels; ++l) {
        const size_t m = size_t(H) >> l, h = m >> 1, per_band = size_t(1) << l;
        for (size_t j = 0; j < level.size(); ++j) {
            level[j].S = S + (band_row + (j >> l) * ((3 * m) >> 1)) * size_t(H) + (j & (per_band - 1)) * m;
        }
        band_row += cv::divUp(level.size(), unsigned(per_band)) * ((3 * m) >> 1);
        // the device table of this level
        std::vector<sum_node_t> table;
        for (const auto &n : level) {
            table.push_back({n.L, n.R, n.S});
        }
        void *device = nullptr;
        if (cudaMalloc(&device, table.size() * sizeof(sum_node_t)) != cudaSuccess) {
            cudaGetLastError();
            why = "DFSCaps: cudaMalloc of a node table failed";
            return false;
        }
        cudaMemcpy(device, table.data(), table.size() * sizeof(sum_node_t), cudaMemcpyHostToDevice);
        set.levels.emplace_back(device, [](void *p) { cudaFree(p); });
        set.counts.push_back(int(level.size()));
        // children: operands L = {A11, A12, A21, A22, S1, S2, S3}, R = {B11, B21, S4, B22, S5, S6, B12}
        std::vector<node_t> next;
        const size_t hl = h * size_t(H);
        for (const auto &n : level) {
            half *Lq[4] = {n.L, n.L + h, n.L + hl, n.L + hl + h};
            half *Rq[4] = {n.R, n.R + h, n.R + hl, n.R + hl + h};
            half *Sb[6] = {n.S, n.S + h, n.S + hl, n.S + hl + h, n.S + 2 * hl, n.S + 2 * hl + h};
            half *Lc[7] = {Lq[0], Lq[1], Lq[2], Lq[3], Sb[0], Sb[1], Sb[2]};
            half *Rc[7] = {Rq[0], Rq[2], Sb[3], Rq[3], Sb[4], Sb[5], Rq[1]};
            for (int i = 0; i < 7; ++i) {
                if (l + 1 == levels) {
                    AA.push_back(Lc[i]);
                    BB.push_back(Rc[i]);
                } else {
                    next.push_back({Lc[i], Rc[i], nullptr});
                }
            }
        }
        level.swap(next);
    }
    CV_Assert(int(AA.size()) == m_leaves);
    set.plan = wgmma_gemm_t::make_plan(AA.data(), H, BB.data(), H, m_N >> m_R, m_leaves, &set);
    return true;
}

bool DFSCaps::initialize(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, int R, cv::cuda::Stream &stream,
                         std::string *why, mode_t mode)
{
    (void) stream;
    std::string reason;
    release();
    auto fail = [&](const std::string &text) {
        if (why) {
            *why = text;
        }
        release();
        return false;
    };
    const int N = A.rows;
    CV_Assert(A.type() == CV_32FC1 && B.type() == CV_32FC1 && A.size() == B.size() && 2 * A.cols == N);
    CV_Assert(A.isContinuous() && B.isContinuous());
    if (R < 2 || R > MAX_RECURSIONS) {
        return fail("DFSCaps needs 2 <= R <= " + std::to_string(MAX_RECURSIONS));
    }
    if (((N >> R) << R) != N) {
        return fail("DFSCaps: N = " + std::to_string(N) + " is not k * 2^R for R = " + std::to_string(R));
    }
    if (!wgmma_gemm_t::available(reason) || !wgmma_gemm_t::supports(N >> R)) {
        return fail("DFSCaps: wgmma n/a (" + (reason.empty() ? std::string("leaf not a multiple of 256") : reason) + ")");
    }
    if (!fits_device_memory(N, R, mode, &reason)) {
        return fail(reason);
    }
    m_N = N;
    m_R = R;
    m_mode = mode;
    m_A = A;
    m_B = B;
    m_leaves = 1;
    for (int l = 1; l < R; ++l) {
        m_leaves *= 7;
    }
    try {
        create_continuous(N, N >> 1, CV_32FC1, m_C);
        m_sets.resize(mode == mode_t::overlapped ? 2 : 1);
        for (auto &set : m_sets) {
            if (!build_set(set, reason)) {
                return fail(reason);
            }
        }
    } catch (const std::exception &e) {
        return fail(std::string("DFSCaps: ") + e.what());
    }
    // targets: leaf (i, q) of sub-problem i, q = its index in the sub-tree's leaf order (base-7
    // digits, the most significant one the child of the sub-root); one W term per level
    m_T = 1 << R;
    CV_Assert(m_T <= wgmma_gemm_t::MAX_REDUCE_TARGETS);
    const auto &W = decoding();
    half *C = reinterpret_cast<half *>(m_C.ptr<float>());
    std::vector<half *> host(size_t(7) * m_leaves * m_T, nullptr);
    for (int i = 0; i < 7; ++i) {
        for (int q = 0; q < m_leaves; ++q) {
            std::vector<int> digits(R);
            digits[0] = i;
            for (int k = R - 1, v = q; k >= 1; --k, v /= 7) {
                digits[k] = v % 7;
            }
            struct place_t { size_t row, col; int sign; };
            std::vector<place_t> places = {{0, 0, 1}};
            for (int k = 0; k < R; ++k) {
                const size_t block = size_t(N) >> (k + 1);
                std::vector<place_t> next;
                for (const auto &p : places) {
                    for (const auto &t : W[digits[k]]) {
                        next.push_back({p.row + t.r * block, p.col + t.c * block, p.sign * t.s});
                    }
                }
                places.swap(next);
            }
            CV_Assert(int(places.size()) <= m_T);
            for (size_t t = 0; t < places.size(); ++t) {
                const auto address = reinterpret_cast<uintptr_t>(C + places[t].row * size_t(N) + places[t].col);
                host[(size_t(i) * m_leaves + q) * m_T + t] = reinterpret_cast<half *>(address | (places[t].sign < 0 ? 1 : 0));
            }
        }
    }
    void *device = nullptr;
    if (cudaMalloc(&device, host.size() * sizeof(half *)) != cudaSuccess) {
        cudaGetLastError();
        return fail("DFSCaps: cudaMalloc of the target lists failed");
    }
    m_targets.reset(device, [](void *p) { cudaFree(p); });
    cudaMemcpy(device, host.data(), host.size() * sizeof(half *), cudaMemcpyHostToDevice);
    if (mode == mode_t::overlapped) {
        int least = 0, greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        if (cudaStreamCreateWithPriority(&m_form_stream, cudaStreamNonBlocking, m_form_priority ? greatest : least)
            != cudaSuccess) {
            cudaGetLastError();
            return fail("DFSCaps: form stream creation failed");
        }
        m_events.resize(1 + 2 * 7); // fork, form step done 0..6, GEMM done 0..6
        for (auto &e : m_events) {
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        }
    }
    m_wgmma.set_round_sync(true); // a launch of 7^(R-1) products is below the automatic barrier's work threshold
    (void) grid_for(1, 1, 1); // the SM count query happens here, not during the capture
    return true;
}

void DFSCaps::run(cv::cuda::Stream &stream, accumulator_t accumulator)
{
    CV_Assert(!m_sets.empty() && m_N > 0);
    if (!m_graph || m_graph_accumulator != accumulator) {
        m_graph = std::make_unique<CaptureGraph>();
        if (!m_graph->start_capture(stream)) {
            CV_Error(cv::Error::GpuApiCallError, "DFSCaps: graph capture failed to start");
        }
        enqueue(stream, accumulator);
        const unsigned long long flags =
            m_mode == mode_t::overlapped && m_form_priority ? cudaGraphInstantiateFlagUseNodePriority : 0;
        if (!m_graph->end_capture(stream, flags)) {
            m_graph.reset();
            CV_Error(cv::Error::GpuApiCallError, "DFSCaps: graph capture failed");
        }
        m_graph_accumulator = accumulator;
    }
    if (!m_graph->launch(stream)) {
        CV_Error(cv::Error::GpuApiCallError, "DFSCaps: graph launch failed");
    }
}

// form step of root product i into one buffer set: its two operands, then the sums level by level
void DFSCaps::form_subtree(set_t &set, int i, cudaStream_t s)
{
    const int N = m_N;
    half *A = reinterpret_cast<half *>(m_A.ptr<float>());
    half *B = reinterpret_cast<half *>(m_B.ptr<float>());
    form_operand(reinterpret_cast<half *>(set.subA.ptr<float>()), A, LEFT[i], N, s);
    form_operand(reinterpret_cast<half *>(set.subB.ptr<float>()), B, RIGHT[i], N, s);
    for (size_t l = 0; l < set.levels.size(); ++l) {
        const unsigned int h = (unsigned(N) >> 2) >> l; // quadrant of a level-l node of the sub-tree: (N/2) >> (l + 1)
        sums_kernel<<<grid_for(h >> 3, h, set.counts[l]), THREADS, 0, s>>>(
            static_cast<const sum_node_t *>(set.levels[l].get()), unsigned(N) >> 1, h);
    }
}

void DFSCaps::enqueue(cv::cuda::Stream &stream, accumulator_t accumulator)
{
    const auto s = (cudaStream_t) stream.cudaPtr();
    const int N = m_N;
    half *A = reinterpret_cast<half *>(m_A.ptr<float>());
    half *B = reinterpret_cast<half *>(m_B.ptr<float>());
    half *C = reinterpret_cast<half *>(m_C.ptr<float>());
    // level l (1 .. R): 2^(l-1) x 2^(l-1) blocks of 2M x 2M, M = N >> l
    const auto transform = [&](auto decode, half *X, half *Y, int l) {
        const unsigned int M = unsigned(N) >> l;
        transform_level_kernel<decltype(decode)::value><<<grid_for(M >> 3, M, size_t(1) << (2 * (l - 1))), THREADS, 0, s>>>(
            X, Y, unsigned(N), M, unsigned(l - 1));
    };
    // 1. ENCODE A and B, every level
    for (int l = 1; l <= m_R; ++l) {
        transform(std::false_type{}, A, B, l);
    }
    cudaMemsetAsync(C, 0, size_t(N) * size_t(N) * sizeof(half), s);
    // 2. the 7 root products, depth first
    m_wgmma.set_stream(stream);
    const bool overlapped = m_mode == mode_t::overlapped;
    const int form_sms = m_form_sms == AUTO_FORM_SMS ? auto_form_sms(N) : m_form_sms;
    m_wgmma.set_persistent_blocks(overlapped && form_sms > 0 ? std::max(1, get_SM_count() - form_sms) : 0);
    half *const *targets = static_cast<half *const *>(m_targets.get());
    if (overlapped) {
        cudaEventRecord(m_events[0], s);
        cudaStreamWaitEvent(m_form_stream, m_events[0], 0); // fork: the form stream joins the capture
    }
    for (int i = 0; i < 7; ++i) {
        auto &set = m_sets[overlapped ? i & 1 : 0];
        if (overlapped) {
            if (i >= 2) {
                cudaStreamWaitEvent(m_form_stream, m_events[1 + 7 + (i - 2)], 0); // its buffer set is free
            }
            form_subtree(set, i, m_form_stream);
            cudaEventRecord(m_events[1 + i], m_form_stream);
            cudaStreamWaitEvent(s, m_events[1 + i], 0); // the GEMM waits for its operands and sums
        } else {
            form_subtree(set, i, s);
        }
        m_wgmma.gemm_reduce(set.plan, targets + size_t(i) * m_leaves * m_T, m_T, N, accumulator);
        if (overlapped) {
            cudaEventRecord(m_events[1 + 7 + i], s);
        }
    }
    // (the form stream rejoined: the last GEMM waited for the last form step)
    // 3. DECODE C, every level
    for (int l = 1; l <= m_R; ++l) {
        transform(std::true_type{}, C, nullptr, l);
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        CV_Error(cv::Error::GpuApiCallError, std::string("DFSCaps: launch failed: ") + cudaGetErrorString(err));
    }
}
