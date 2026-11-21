/**
 * @file bfs_caps.hpp
 * @brief Host side of the CAPS GEMM: class BFS (recursion tree buffers, pass launchers, leaf GEMM)
 * @author GEMM Project
 * @date 2025
 *
 * CAPS here: Strassen-Winograd in an alternative basis, evaluated breadth first over the recursion
 * tree (all nodes of a level together, one cooperative kernel per pass, see src/bfs_caps.cu). One
 * C = A * B of N x N half matrices (N a power of two) with R recursions runs as
 *   1. ENCODE      A and B are transformed into the alternative basis, in place (all R levels);
 *   2. BILINEAR    every internal node (levels 0 .. R - 1) gets the six quadrant sums S1 .. S6
 *                  (the operands of its children m3, m5, m6, m7) in its S area (m_S);
 *   3. leaf GEMM   the 7^R leaf products of (N >> R) x (N >> R) blocks as one batched GEMM,
 *                  batched_leaf() (our mma.sync kernel, cuBLAS, or our wgmma kernel on Hopper),
 *                  into the product buffers;
 *   4. RECONSTRUCT the 7 products of each node are combined into the node's product, which
 *                  overwrites its S area; level R - 1 first, then up to the root;
 *   5. DECODE      the root product (the result, get_result()) is transformed back, in place.
 * forward_pass() = 1 + 2, backward_pass() = 4 + 5.
 *
 * Data layout: all matrices hold half values but are typed CV_32FC1 (one float = two halves), so a
 * GpuMat of c columns is 2c halves wide, and pointers are float * on the host. A, B and m_S share
 * one row pitch of N halves (N / 2 floats); every node's operands and S area are views into them
 * with that pitch, so a kernel addresses any node with (pointer, global leading dimension) alone.
 *
 * Node order (address caches, constant memory, pointer arrays): breadth first, level by level, and
 * within a level by parent, then child m1 .. m7; level l starts at index (7^l - 1) / 6.
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <list>
#include <vector>
#include <map>
#include <string>
#include <opencv2/core/base.hpp>

#ifndef Q_MOC_RUN // Qt's moc cannot parse CUDA 13 Thrust (CCCL 3 namespace macros)
#include <thrust/device_vector.h>
#endif
#include <cublas_gemm.hpp>
#include <wgmma_gemm.hpp>

#include <core.hpp>
#include <cuda_fp16.hpp>
#include <cublas_api.h>
#include <vector_types.h>

/**
 * @class BFS
 * @brief CAPS GEMM: owns the recursion tree buffers of one (N, R) and launches its passes
 * 
 * initialize() lays out the tree for the given A and B (it keeps their headers, the passes
 * overwrite them), the pass wrappers below launch src/bfs_caps.cu kernels that read the node addresses
 * from constant memory (store_address_cache()), and the batched_* functions run the leaf GEMM.
 * Each step is asynchronous on the given stream; the caller orders and synchronizes them.
 * 
 * @note NOT thread-safe: the node address tables live in __constant__ memory, which is global to
 *       the module, so only one BFS object per process can be in use at a time (the last
 *       initialize() wins).
 * @warning initialize() of a second object silently redirects the kernels of the first one.
 * 
 * R is the number of recursions: 0 = plain GEMM (one cuBLAS HGEMM), 1 = one Strassen-Winograd
 * level (7 products of N/2 blocks), ..., R: 7^R products of N/2^R blocks (1 <= R <= MAX_RECURSIONS).
 *
 * @example
 * ```cpp
 * BFSCaps bfs;
 * bfs.initialize(A, B, R, stream);
 * bfs.forward_pass(stream);
 * bfs.batched_leaf(stream, wgmma_gemm_t::accumulator_t::half);
 * bfs.backward_pass(stream);
 * ```
 */
class BFSCaps
{
public:
    /** @brief Warps along x in a block of bfs_kernel (blockDim.x = WARPS_IN_X_DIR * 32);
     *  also part of its __launch_bounds__ */
    static constexpr auto WARPS_IN_X_DIR = 1;
    /** @brief Warps along y in a block of bfs_kernel (blockDim.y): CMake BFS_WARPS_Y (default 4: 1 x 4 warps = 128 threads). Each virtual block covers
     *  WARPS_IN_Y_DIR rows; a virtual block may straddle nodes (node_idx is per thread). The tuned
     *  loop counts depend on it. */
    static constexpr auto WARPS_IN_Y_DIR = BFS_WARPS_Y;
    static_assert(WARPS_IN_Y_DIR >= 1 && WARPS_IN_Y_DIR <= 32 && (WARPS_IN_Y_DIR & (WARPS_IN_Y_DIR - 1)) == 0,
                  "BFS_WARPS_Y must be a power of two from 1 to 32 (blocks of at most 1024 threads)");

    /**
     * @brief Pass flags: the template argument of caps<op>() / process<op>()
     * @details Each flag enables one stage of bfs_kernel (src/bfs_caps.cu); a combination runs the
     * stages in the order encode, bilinear, reconstruct, decode inside one cooperative launch,
     * separated by grid barriers. Every combination used needs its own kernel instantiation
     * (select_kernel() in src/bfs_caps.cu); process() accepts values 1 .. 127.
     * @note These values can be combined using bitwise OR operations
     */
    enum {
        ENCODE = 1,                    ///< Transform A and B into the alternative basis, in place
        BILINEAR = 2,                  ///< Form the sums S1 .. S6 of every internal node (the bilinear, i.e. forward, phase of Strassen; no interpolation)
        RECONSTRUCT_LEAF_MULTS = 4,    ///< Rebuild the parents of the leaves (level R - 1) from the leaf products
        RECONSTRUCT_INTERMEDIATE_MULTS = 8, ///< Rebuild the intermediate nodes, layer by layer up to the root
        DECODE = 16,                   ///< Transform the result (root S area) back from the alternative basis, in place
        FULL_RECONSTRUCT = RECONSTRUCT_LEAF_MULTS | RECONSTRUCT_INTERMEDIATE_MULTS, ///< Both: the backward pass up to the root, without DECODE
        FORWARD = ENCODE | BILINEAR,   ///< Combined forward pass
        BACKWARD = FULL_RECONSTRUCT | DECODE, ///< Combined backward pass
    };

    /**
     * @brief Row loop counts of the four passes of bfs_kernel
     * @details A pass with loop count L walks its matrices in L row slices (row r handles rows
     * r, r + L, r + 2L, ... of every level, N / L rows of the top level and (N >> l) / L of level
     * l), so the data one slice touches can stay in L2. Valid counts are powers of two in
     * [1, N >> R] (valid_loops() in src/bfs_caps.cu). The values come from gen/tuning_table.hpp
     * (tuning::bfs_loops()) when it has an entry for this GPU or "default", else from the L2
     * heuristic of initialize(); with LOOPS_COUNT_AS_TEMPLATE_ARGS=ON the kernel uses the table's
     * compiled-in values and these fields are not passed to it.
     */
    struct config_t
    {
        unsigned int m_transform_A_and_B_loops_count = 1;  ///< Loop count of ENCODE (A and B into the alternative basis)
        unsigned int m_transform_C_loops_count = 1;       ///< Loop count of DECODE (the result back)

        unsigned int m_bilinear_loops_count = 1;          ///< Loop count of BILINEAR (the sums)
        unsigned int m_bilinear_rows_per_loop;            ///< N / m_bilinear_loops_count; informational, not passed to the kernel

        unsigned int m_reconstruct_loops_count = 1;       ///< Loop count of RECONSTRUCT_LEAF_MULTS / _INTERMEDIATE_MULTS
        unsigned int m_reconstruct_rows_per_loop = 1;     ///< (N >> R) / m_reconstruct_loops_count (leaf rows); informational

        /// Default of m_grid_divisor (what process() always used; the tuner used 3 before)
        static constexpr unsigned int DEFAULT_GRID_DIVISOR = 4;
        /// Cooperative grid of every bfs_kernel launch (process() and the tuner): the co-resident
        /// blocks of the kernel (blocks per SM x SMs) / m_grid_divisor; 1 = all of them. Only the
        /// speed depends on it (the kernels grid-stride over their virtual blocks): more blocks
        /// hide more DRAM latency, fewer leave less idle work in the last round of a level.
        unsigned int m_grid_divisor = DEFAULT_GRID_DIVISOR;

        /**
         * @brief Heuristic loop counts: halve the estimated working set of a pass until it fits
         *        into L2 + 8%, doubling the loop count each time (see src/bfs_caps.cpp)
         * @param N Matrix dimension (N x N matrices)
         * @param R Number of recursions
         */
        void initialize(int N, int R);
    };

    /** @brief Host element type of the buffers: float, i.e. a pair of halves (see the file comment) */
    typedef float T;
    
    /**
     * @struct layer_info_t
     * @brief One internal node of the tree: its two operands and its S area
     * @details 24 bytes, stored as an array in __constant__ memory (__intermediate_address_cache__
     * in src/bfs_caps.cu) in node order. All three pointers use the global pitch of N halves. For the
     * root m_L / m_R are A and B; for a child they point into the parent's A/B quadrants or the
     * parent's S1 .. S6 (e.g. m3 = A21 * S4). m_S is the node's S area (3M/2 rows of an M x M
     * node): first the sums S1 .. S6, after the backward pass the node's product in its top M rows.
     */
    struct layer_info_t
    {
        /**
         * @brief Takes the data pointers of three views and checks that they fit together
         * @param A Left operand, M rows x M / 2 floats (M x M halves)
         * @param B Right operand, same size and type as A
         * @param S The node's S area: 3M/2 rows, as wide as A
         * @note All three must have the same step (the global pitch), which the kernels assume.
         *       m_L / m_R are taken at row 0 of A / B; m_S at the start of S.
         */
        layer_info_t(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::GpuMat &S)
            : m_L(A.ptr<float>(0))
            , m_R(B.ptr<float>(0))
            , m_S(S.ptr<float>())
        {
            CV_Assert(0 == A.step % sizeof(T));
            CV_Assert(0 == B.step % sizeof(T));
            CV_Assert(0 == S.step % sizeof(T));
            CV_Assert(A.cols == S.cols);
            CV_Assert(3 * A.rows / 2 == S.rows);
            CV_Assert(A.type() == B.type());
            CV_Assert(A.size() == B.size());
            CV_Assert(A.type() == S.type());
            CV_Assert(A.step == B.step);
            CV_Assert(A.step == S.step);
        }
        layer_info_t() = default;
        const T* m_L; ///< Left operand (top-left element)
        const T* m_R; ///< Right operand (top-left element)
        T* m_S; ///< The node's S area: sums S1 .. S6, later the node's product
    };

    /**
     * @struct mult_buffers_t
     * @brief Device pointer arrays of the batched leaf GEMM (prepare_mult_buffers())
     * @details 7^R entries each, entry 7 * j + i = product m(i+1) of node j of level R - 1 (node
     * order): C[e] = A[e] * B[e], row-major (N >> R) x (N >> R) half matrices.
     */
    struct mult_buffers_t
    {
        int ld_ab = 0; ///< Leading dimension of the operands, in halves: N (they live in A, B or m_S)
        int ld_c = 0; ///< Leading dimension of the products, in halves: N >> R (dense buffers)
        thrust::device_vector<half *> AA; ///< Left operands (A quadrants or S1, S2, S3 of the node)
        thrust::device_vector<half *> BB; ///< Right operands (B quadrants or S4, S5, S6 of the node)
        thrust::device_vector<half *> CC; ///< Device vector of C matrix pointers (entry p: leaf product p, tree order)
        /// LEAF_GEMM=wgmma only: tensor maps of AA / BB (prepare_leaf_plan(); built outside any
        /// graph capture, bound to the addresses in AA / BB). Empty with the other leaf GEMMs.
        wgmma_gemm_t::plan_t wgmma_plan;
    };

    /**
     * @brief Quadrant (i, j) of a dimension x dimension half matrix given by its pointer
     * @param ptr Top-left element of the matrix (float = two halves)
     * @param dimension Matrix size in halves
     * @param i Block row index, 0 or 1
     * @param j Block column index, 0 or 1
     * @param global_lds Row pitch in floats (N / 2 for every buffer of the tree)
     * @return View of H = dimension / 2 rows x W = H / 2 floats (H x H halves) at row i * H,
     *         float column j * W
     */
    static auto block(const float *ptr, int dimension, int i, int j, int global_lds) {
        const auto H = dimension / 2;
        const auto W = H / 2;
        return cv::cuda::GpuMat(H,
                                W,
                                CV_32FC1,
                                const_cast<float *>(ptr + size_t(i) * H * global_lds + j * W),
                                global_lds * sizeof(float));
    };

    /**
     * @brief Block (i, j) of an S area: W = A.cols / 2 floats wide, H = 2W rows (a square of H x H
     *        halves); used to cut S1 .. S6 out of a 3H x 2W float area (i = 0..2, j = 0..1)
     * @param A Input matrix (float view of half data)
     * @param i Block row index
     * @param j Block column index
     * @return GpuMat view of the specified block
     */
    static inline auto block(const cv::cuda::GpuMat& A, int i, int j)
    {
        const auto W = A.cols >> 1;
        const auto H = W << 1;
        return A.rowRange(i * H, (i + 1) * H).colRange(j * W, (j + 1) * W);
    }

public:
    /**
     * @brief Rows of the S buffer (m_S) that initialize(N, R) allocates.
     *
     * m_S is one float matrix of N / 2 columns, i.e. N halves per row (the width of A). It holds,
     * for every node of levels 0 .. R - 1 (1 + 7 + ... + 7^(R-1) nodes), the node's S area: the six
     * quadrant-sized sums S1 .. S6 laid out as
     *
     *     S1 S2        S1, S2, S3: sums of A quadrants, S4, S5, S6: sums of B quadrants
     *     S3 S4        (the operands of m3, m5, m6, m7, see initialize())
     *     S5 S6
     *
     * so a node of size M x M halves takes M halves x 3M/2 rows (6 (M/2)^2 halves). The same area
     * later takes the node's reconstructed product (M x M = S1 .. S4); the root's is the result C
     * (get_result()), so R >= 1 needs no separate C buffer.
     *
     * The areas are packed by find_roi() (first fit in a list of free rectangles): the root fills
     * the full width, the narrower areas of deeper levels are placed side by side, and new rows are
     * appended only when no free rectangle is tall enough. Hence rows >= sum over l < R of
     * 7^l * 6 * (N / 2^(l+1))^2 halves / N; the packing loses 0 .. 9% (N = 16384: R = 1 exact,
     * R = 2 2304 MB instead of 2112 MB, R = 4 8640 MB instead of 8580 MB).
     */
    static int sums_rows(int N, int R);

    /**
     * @brief Device bytes of a CAPS run of N x N half matrices with R recursions.
     *
     * The sum of
     *  - A and B: 2 N^2 halves, allocated by the caller but counted, so that the check can be done
     *    before anything is allocated (the encoder overwrites them in place, no copies);
     *  - S: sums_rows(N, R) x N / 2 floats (all sums of levels 0 .. R - 1, later the products of
     *    those nodes and the result, see sums_rows());
     *  - the leaf products: one continuous allocation per node of level R - 1 (7^(R-1) of them),
     *    each 7 products of leaf x leaf halves (leaf = N >> R), rounded up to 2 MB because the CUDA
     *    driver hands out large allocations in 2 MB pages;
     *  - the pointer arrays of the batched leaf GEMM: 3 x 7^R device pointers (A, B, C of every
     *    product; 56 KB at R = 4).
     * R = 0 (plain GEMM): A, B and the N x N result m_C.
     *
     * Not counted (covered by reserved_memory()): the CUDA context, the cuBLAS workspace, CUDA graphs, the allocator's rounding of
     * the small allocations.
     *
     * Examples (MB): N = 16384: R = 1 2688, R = 2 4896, R = 3 8376, R = 4 14466;
     * N = 32768: R = 1 10752, R = 2 19584 (A + B alone are 4096).
     */
    static size_t required_memory(int N, int R);

    /**
     * @brief Device memory taken by things required_memory() does not count; a heuristic margin.
     *
     * A fixed 2 GiB, whatever the device size (@p total is not used; it was 5% of the device, at
     * least 512 MB, i.e. 4 GB on an 80 GB H100). Meant for the CUDA context (a few hundred MB,
     * already allocated when fits_device_memory() runs), the cuBLAS workspace (MBs to tens of MB
     * per handle), CUDA graphs and the small pointer arrays. It does not cover memory of other
     * processes (desktop, browser, other CUDA applications), see fits_device_memory().
     */
    static size_t reserved_memory(size_t total);

    /**
     * @brief Whether a CAPS run of (N, R) fits into this GPU.
     *
     * True when required_memory(N, R) + reserved_memory(total) <= total device memory
     * (cudaMemGetInfo()). Deliberately against the total, not the free memory: the answer does not
     * depend on what this or another process holds at the moment, so tests and the tuner skip the
     * same (N, R) on every run, and a check made after the caller allocated A and B gives the same
     * answer (initialize() checks again). The price: memory taken by other processes (on Windows
     * the desktop takes 1-2 GB) can still make the allocation in initialize() fail; it then returns
     * false and releases everything ("out of memory").
     *
     * @param why if not null and the run does not fit: "N=.., R=.. needs X MB + Y MB reserved, the
     *            GPU has Z MB", or that cudaMemGetInfo() failed (no CUDA device)
     */
    static bool fits_device_memory(int N, int R, std::string *why);

    /**
     * @brief Prepare a run of C = A * B: allocate and lay out the tree, set m_config, upload the
     *        node addresses to constant memory
     * @details Frees the buffers of a previous initialize() first. Only the headers of A and B are
     * kept: the passes read and overwrite their data (ENCODE works in place), so the caller keeps
     * them alive and re-uploads the inputs before another run.
     * @param A Left input matrix: N x N halves as an N x N / 2 CV_32FC1 GpuMat, N a power of two,
     *          continuous (step = N halves); B the same
     * @param B Right input matrix
     * @param R Number of recursions, 0 .. MAX_RECURSIONS (0: plain cuBLAS GEMM)
     * @param stream CUDA stream for operations
     * @return true if initialization successful, false otherwise (also when (N, R) does not fit into
     *         this GPU, see fits_device_memory(); nothing stays allocated then)
     */
    bool initialize(
        const cv::cuda::GpuMat& A,
        const cv::cuda::GpuMat& B,
        int R,
        cv::cuda::Stream& stream);

    /**
     * @brief Launch the passes selected by @p operation on the tree of the last initialize()
     * @tparam operation Combination of the pass flags (ENCODE, BILINEAR, FORWARD, BACKWARD, etc.)
     * @param stream CUDA stream for operations
     * @details Forwards to process<operation>(m_N, m_R): one cooperative launch of bfs_kernel.
     * Nothing to do for R = 0 (plain GEMM, see batched_cublas()).
     */
    template<int operation>
    void caps(cv::cuda::Stream &stream)
    {
        if (0 == m_R) {
            return; // no recursion: nothing to encode, sum, reconstruct or decode
        }
        process<operation>(m_N, m_R, stream);
    }

    /** @brief ENCODE only: A and B into the alternative basis, in place (minimum block size 64x64) */
    void encode(cv::cuda::Stream &stream) { caps<BFSCaps::ENCODE>(stream); }
    /** @brief BILINEAR only: the sums S1 .. S6 of all internal nodes (needs ENCODE done) */
    void bilinear(cv::cuda::Stream &stream) { caps<BFSCaps::BILINEAR>(stream); }
    /** @brief ENCODE + BILINEAR: everything before the leaf GEMM */
    void forward_pass(cv::cuda::Stream &stream) { caps<BFSCaps::FORWARD>(stream); }
    /** @brief Reconstruction of all levels + DECODE: everything after the leaf GEMM; the result is
     *  then in get_result() */
    void backward_pass(cv::cuda::Stream &stream) { caps<BFSCaps::BACKWARD>(stream); }

    /** @brief The reconstruction of levels R - 1 .. 0 without DECODE */
    void reconstruct(cv::cuda::Stream &stream) { caps<BFSCaps::FULL_RECONSTRUCT>(stream); }
    /** @brief DECODE only: the root product back from the alternative basis, in place */
    void decode(cv::cuda::Stream &stream) { caps<BFSCaps::DECODE>(stream); }

    /** @brief Frees the buffers (GpuMat members release themselves) */
    ~BFSCaps();

    /** @brief S area views of all internal nodes, per level, in node order */
    auto& get_intermediate_sums() { return m_intermediate_sums; }
    /** @brief layer_info_t of all internal nodes, per level (host copy of the constant memory table) */
    auto& get_intermediate_address_cache() { return m_intermediate_address_cache; }
    /** @brief Product buffer of each level R - 1 node (7 leaf products each), indexed [level] */
    auto& get_mult_address_cache() { return m_mult_address_cache; }

    /**
     * @brief One leaf product m_i of a level R - 1 node: operands L, R (views into A, B or the
     *        node's S area, pitch N halves) and its slot C in the node's product buffer (pitch leaf)
     */
    struct leaf_product_t
    {
        cv::cuda::GpuMat L, R, C;
    };
    /**
     * @brief The 7^R leaf products of the tree built by initialize(N, R), entry 7 * j + i =
     *        product m_(i + 1) of node j of level R - 1 (the order of the batched leaf GEMM)
     */
    std::vector<leaf_product_t> leaf_products(int N, int R) const;

    /**
     * @brief Download intermediate results (testing purposes only)
     * @details Blocking: waits for @p stream. Meaningful after BILINEAR and before the reconstruction
     * overwrites the S areas.
     * @param HS Output map: HS[level][node] = {S1, ..., S6} of that node (half data in CV_32FC1)
     * @param stream CUDA stream for operations
     */
    void download_intermediate_sums(std::map<int, std::vector<std::vector<cv::Mat>>> &HS,
                                    cv::cuda::Stream &stream);

    /**
     * @brief Download intermediate multiplications (testing purposes only)
     * @details Blocking. After the reconstruction: the product of every internal node, i.e. the top
     * 2/3 of its S area (M x M halves).
     * @param HS Output vector: HS[level][node]
     * @param stream CUDA stream for operations
     */
    void download_intermediate_mults(std::vector<std::vector<cv::Mat>> &HS,
                                     cv::cuda::Stream &stream);

    /** @brief The result C as an N x N half view (N x N / 2 CV_32FC1, pitch N halves): the top N
     *  rows of the root S area, valid after backward_pass(); R = 0: m_C.
     *  A view, not a copy: the next run overwrites it. */
    cv::cuda::GpuMat get_result();

    /**
     * @brief Download leaf multiplications (testing purposes only)
     * @details Blocking. Valid after the leaf GEMM and before the leaf reconstruction.
     * @param HM Output vector, one entry per node in node order with its 7 products m1 .. m7; only
     *           the level R - 1 nodes own product buffers, the other entries hold empty Mats
     * @param stream CUDA stream for operations
     */
    void download_leaf_mults(std::vector<std::vector<cv::Mat>>& HM,
                             cv::cuda::Stream& stream);

    // should be private in production code
// private:

    /**
     * @brief Initialize BFS with input matrices (internal version)
     * @details Allocates m_S, cuts the S area of every internal node out of it (find_roi(), the
     * same packing as sums_rows()), allocates one continuous 7-product buffer per level R - 1 node,
     * fills the per-level address caches and sets m_config (heuristic, then the tuning table).
     * Does not upload anything; the public initialize() does that.
     * @param A Left input matrix
     * @param B Right input matrix
     * @param R Number of recursions, >= 1 (R <= 0 returns false)
     * @return true if initialization successful, false otherwise; allocation failures throw
     */
    bool initialize(const cv::cuda::GpuMat &A,
                    const cv::cuda::GpuMat &B,
                    int R); // recursions

    /**
     * @brief Fill m_mult_buffers: for each level R - 1 node its 7 leaf products (operands, output)
     * @param N Matrix dimension
     * @param R Number of recursions (>= 1)
     * The pointers come from leaf_products(N, R).
     */
    void prepare_mult_buffers(int N, int R);

    /**
     * @brief Allocate m_S: sums_rows(N, R) rows of N / 2 floats (the S areas are cut out later)
     * @param N Matrix dimension
     * @param R Number of recursions
     */
    void create_intermediate_sums_layout(int N, int R);

    /**
     * @brief One cooperative launch of bfs_kernel<operation, N, R> (src/bfs_caps.cu)
     * @details The kernel is looked up in the instantiated table (throws when (N, R) has none);
     * the loop counts come from m_config unless they are compiled in. Reads the node addresses
     * from constant memory, so the last initialize() must be this object's.
     * @tparam operation Combination of pass flags
     * @param N Matrix dimension
     * @param R Number of recursions
     * @param stream CUDA stream for operations
     */
    template<int operation>
    void process(int N, // dim
                 int R, // recursions
                 cv::cuda::Stream &stream);

    /**
     * @brief Time every valid loop count of the pass @p operation and keep the fastest
     * @details Only with LOOPS_COUNT_AS_TEMPLATE_ARGS=OFF (otherwise prints a note and returns
     * m_config). Runs the kernels on the input matrices and overwrites them.
     * @param operation One pass flag: ENCODE, BILINEAR, FULL_RECONSTRUCT or DECODE
     * @param warm_ups Untimed launches per candidate
     * @param iterations Timed launches per candidate
     * @return m_config with the best loop count of that pass (rows per loop updated)
     */
    config_t select_best_options(unsigned int operation, int warm_ups, int iterations);

    /**
     * @brief Tune the loop counts of ENCODE, BILINEAR, FULL_RECONSTRUCT and DECODE.
     * @details Needs a build with LOOPS_COUNT_AS_TEMPLATE_ARGS=OFF (loop counts are kernel
     * arguments only then). Prints a select_kernel() table line to std::cout; with @p apply
     * the result becomes the configuration used by process(). Overwrites the input matrices.
     */
    config_t tune(int warm_ups, int iterations, bool apply);

    /** @brief The loop counts process() uses (see config_t) */
    const auto &get_config() const { return m_config; }
    /** @brief Replace the configuration process() uses (e.g. one returned by tune()) */
    void set_config(const config_t &config) { m_config = config; }
    /** @brief Cooperative grid = co-resident blocks / divisor for every launch (config_t::m_grid_divisor) */
    void set_grid_divisor(unsigned int divisor)
    {
        CV_Assert(divisor >= 1);
        m_config.m_grid_divisor = divisor;
    }

    /** @brief Matrix size N of the last initialize() */
    auto get_dimension() const { return m_N; }
    /** @brief Number of recursions R of the last initialize() (0: plain GEMM) */
    auto get_recursions() const { return m_R; }

    /** @brief The leaf GEMM with cuBLAS: all 7^R products of m_mult_buffers in one batched HGEMM
     *  (cublas_gemm_t::gemm(), half accumulation); R = 0: one plain GEMM A * B into m_C */
    void batched_cublas(cv::cuda::Stream &stream);
    /** @brief Execute batched CUBLAS GEMM operations with specific algorithm and accumulation
     *  (CUBLAS_COMPUTE_16F: half, CUBLAS_COMPUTE_32F: float; see cublas_gemm_t::gemm_ex()) */
    void batched_cublas_ex(cv::cuda::Stream &stream,
                           cublasGemmAlgo_t algo,
                           cublasComputeType_t compute);
    /** @brief The leaf GEMM of the pipeline chosen by CMake LEAF_GEMM over all
     *  7^R products: leaf_gemm(m_mult_buffers); R = 0: one plain GEMM (batched_cublas_ex() with the
     *  matching compute type) */
    void batched_leaf(cv::cuda::Stream &stream,
                      wgmma_gemm_t::accumulator_t accumulator);
    /** @brief One batched leaf GEMM over the products of @p buffers (leaf x leaf each, leaf =
     *  N >> R) with the leaf GEMM chosen by CMake LEAF_GEMM: cuBLAS
     *  (cublasGemmBatchedEx, 16F / 32F compute matching @p accumulator) or our wgmma kernel
     *  (Hopper; needs buffers.wgmma_plan from prepare_leaf_plan()).
     *
     *  The generic entry for any subset of the leaf products: batched_leaf() passes all 7^R
     *  (m_mult_buffers). Requires R >= 1. */
    void leaf_gemm(cv::cuda::Stream &stream,
                   mult_buffers_t &buffers,
                   wgmma_gemm_t::accumulator_t accumulator);
    /** @brief Per-batch preparation of the leaf GEMM, after @p buffers got their pointers and
     *  leading dimensions: LEAF_GEMM=wgmma encodes and uploads the tensor maps
     *  (wgmma_gemm_t::make_plan(), synchronous, so never inside a graph capture); the other leaf
     *  GEMMs need nothing. initialize() calls it for m_mult_buffers. Returns false (with a message) when the GPU or the build cannot run wgmma. */
    bool prepare_leaf_plan(mult_buffers_t &buffers);


    /**
     * @brief Store address cache in GPU constant memory
     * @details Asynchronous copies on @p stream into __intermediate_address_cache__ and
     * __mult_address_cache__ (src/bfs_caps.cu); both tables are checked against their capacity
     * (ADDRESS_CACHE_SIZE, MULT_CACHE_SIZE) first.
     * @param intermediate_cache All internal nodes in node order (1 + 7 + ... + 7^(R-1))
     * @param mult_cache Product buffer of each level R - 1 node, in node order (7^(R-1))
     * @param stream CUDA stream for operations
     * @return true if successful, false otherwise
     */
    bool store_address_cache(const std::vector<BFSCaps::layer_info_t> &intermediate_cache,
                             const std::vector<float *> &mult_cache,
                             cv::cuda::Stream &stream);

// some public members
public:
    mult_buffers_t m_mult_buffers; ///< Pointer arrays of the batched leaf GEMM
    cublas_gemm_t m_cgemm; ///< cuBLAS handle wrapper (batched_cublas(), batched_cublas_ex())
    wgmma_gemm_t m_wgmma_gemm; ///< our Hopper TMA + wgmma GEMM (leaf_gemm() with LEAF_GEMM=wgmma)

private:
    int m_N = 0; ///< matrix size of the last initialize()
    int m_R = 0; ///< recursions of the last initialize()
    cv::cuda::GpuMat m_C; ///< result of the plain GEMM (R = 0 only)

    config_t m_config; ///< Configuration settings for BFS operations
    std::vector<std::list<cv::cuda::GpuMat>> m_intermediate_sums; ///< [level]: S area view of each node (views into m_S; used by get_result() and the downloads) (TODO: Remove)
    std::vector<std::list<cv::cuda::GpuMat>> m_leaf_multiplications; ///< [level]: product buffer of each node (owning; empty except at level R - 1)
    std::vector<std::list<layer_info_t>> m_intermediate_address_cache; ///< [level]: operands and S area of each node
    std::vector<std::list<float*>> m_mult_address_cache; ///< [level]: product buffer pointers; only [R - 1] is filled
    cv::cuda::GpuMat m_S; ///< All S areas of levels 0 .. R - 1, packed (sums_rows() x N / 2 floats)
    cv::cuda::GpuMat m_M; ///< Multiplication matrix (not used by the current pipeline)
    cv::cuda::GpuMat m_A, m_B; ///< Headers of the caller's A and B (shared data, not copies)
    cv::cuda::Stream m_streams[ARITHEMTIC_OPS_COUNT]; ///< CUDA streams for parallel operations (legacy, unused)
    cv::cuda::Event m_start_events[ARITHEMTIC_OPS_COUNT]; ///< Start timing events (legacy, unused)
    cv::cuda::Event m_end_events[ARITHEMTIC_OPS_COUNT]; ///< End timing events (legacy, unused)
};
