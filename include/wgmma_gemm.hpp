/**
 * @file wgmma_gemm.hpp
 * @brief Batched HGEMM for Hopper (sm_90a): TMA + mbarrier pipeline + wgmma (src/wgmma_gemm.cu),
 *        the H100 leaf GEMM of the CAPS pipeline (CMake LEAF_GEMM=wgmma)
 *
 * For every batch entry i, row-major C[i] = A[i] * B[i],
 * M x M half matrices, alpha = 1, beta = 0, accumulation in half (wgmma .f16) or float
 * (wgmma .f32, rounded to half once in the epilogue).
 *
 * Why a separate kernel: on sm_90 the mma.sync path (mma_gemm_t) reaches only about half of the
 * tensor core peak. Full rate needs the asynchronous warpgroup MMA (wgmma.mma_async), which reads
 * both operands straight from shared memory through 64-bit "matrix descriptors", and the Tensor
 * Memory Accelerator (TMA, cp.async.bulk.tensor), which copies whole 2D boxes global -> shared
 * (with the 128 B swizzle that wgmma expects) without using registers or threads.
 *
 * TMA needs a tensor map (CUtensorMap, 128 B, encoded on the host by the driver function
 * cuTensorMapEncodeTiled) for every global matrix it reads. Our operands are not one strided 3D
 * tensor (they are quadrants of A and B or S areas of the tree, at irregular addresses), so every
 * batch entry gets its own pair of maps (A[i], B[i]). Encoding them takes host work and a
 * host -> device copy, which must not happen inside a CUDA graph capture (DFSCaps records the
 * whole pipeline) nor once per call (launch overhead). Hence the split:
 *   1. plan_t make_plan(A, lda, B, ldb, dimension): once, at initialize time (BFSCaps::initialize(),
 *      DFSCaps::initialize()). Encodes 2 * count maps and uploads them.
 *   2. gemm(plan, C, ldc, accumulator): per call, only a kernel launch (capture safe).
 * The plan is bound to the operand addresses: re-plan whenever A[i] / B[i] change.
 *
 * Requirements (checked by make_plan() / gemm(), cv::Exception otherwise):
 *   - a GPU of compute capability 9.0 and a build that contains sm_90a code (CMake
 *     CUDA_ARCH=90a-real or "89;90a-real"; the "a" is required: wgmma and setmaxnreg are
 *     architecture-specific features that plain sm_90 / compute_90 do not have);
 *   - M a multiple of 256 (tile BN; BM = 128 and BK = 64 divide it then);
 *   - A[i], B[i] 16 B aligned, lda and ldb multiples of 8 halfs (TMA: 16 B aligned rows);
 *   - C[i] 4 B aligned and ldc even (half2 stores).
 * Use available() to ask without exceptions (e.g. to skip a test on a 4090).
 *
 * The compiled interface takes raw pointers only (thrust ABI namespaces differ between nvcc and
 * the host compiler); the thrust overloads are inline.
 */

#pragma once

#include <cuda_fp16.h>
#ifndef Q_MOC_RUN // Qt's moc cannot parse CUDA 13 Thrust (CCCL 3 namespace macros)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <opencv2/core/cuda.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * @brief Batched M x M HGEMM C[i] = A[i] * B[i] with TMA and wgmma (Hopper only). Holds only the
 *        launch stream; the per-operand state lives in plan_t objects.
 */
struct wgmma_gemm_t
{
    /// Accumulators of the tensor cores: half (wgmma .f16, stored as is) or single (.f32, rounded once)
    enum class accumulator_t { half, single };

    /// Block tile and pipeline (fixed for now; src/wgmma_gemm.cu explains the choice)
    static constexpr int BM = 128;     ///< rows of C per block: 2 consumer warpgroups x 64
    static constexpr int BN = 256;     ///< columns of C per block: one wgmma m64n256k16 wide
    static constexpr int BK = 64;      ///< k slice per pipeline stage: 64 halfs = one 128 B swizzle row
    static constexpr int STAGES = 4;   ///< shared memory ring depth (4 x 48 KB)

    /**
     * @struct plan_t
     * @brief Tensor maps of one batch of operands (make_plan()); cheap to copy (shared ownership
     *        of the device array)
     */
    struct plan_t
    {
        std::shared_ptr<void> maps; ///< device: 2 * count CUtensorMap (A[i] at 2i, B[i] at 2i + 1)
        /// device int: the round barrier's counter (persistent schedule, cluster 1), reset by gemm()
        /// before every launch that uses it. So one plan
        /// must not run on two streams at once.
        std::shared_ptr<void> counter;
        int count = 0;              ///< batch entries
        int dimension = 0;          ///< M of every product
        const void *key = nullptr;  ///< identity of the operand pointer array it was built from (for checks)
        bool valid() const { return maps != nullptr && count > 0; }
    };

    wgmma_gemm_t() = default; ///< launches on the default stream until set_stream()

    /// Stream of all following launches; only the raw cudaStream_t is kept
    void set_stream(cv::cuda::Stream &stream) { m_stream = (cudaStream_t)stream.cudaPtr(); }

    /// Tile order: which tiles of a product run at the same time, for L2 reuse. Groups of g tile
    /// rows (g >= 1), column by column within a group (Triton's grouped ordering, like the g > 0
    /// orders of mma_gemm_t): a wave of blocks then covers about g tile rows x (wave / g) tile
    /// columns, so A row panels and B column panels are both reused from L2. g = 1 is row by row
    /// (the first version): with 256 wide tiles a 132-block wave of a 16384 leaf covers only 2
    /// tile rows x all 64 tile columns, and every tile row reads B from DRAM again (leaf 8192 and
    /// up ran at half of cuBLAS's speed). The products of a batch still run one after the other.
    /// 0 (the default) chooses g automatically, per schedule (set_schedule()), from the H100
    /// sweeps: persistent 4 from 32 tile columns on (N >= 8192), else whole tile columns;
    /// one tile per block max(8, tile columns / 2). See wgmma_gemm_t::gemm() in src/wgmma_gemm.cu.
    static constexpr int AUTO_ORDER = 0;
    static constexpr int DEFAULT_ORDER = AUTO_ORDER;
    void set_order(int g) { m_order = g; }
    int order() const { return m_order; }
    /// Which blocks compute which tiles (src/wgmma_gemm.cu): persistent = one block per SM, each
    /// works through every gridDim.x-th tile of the grouped order (a stable L2 working set, and the
    /// next tile's loads overlap the current tile's stores); one_tile_per_block = one block per
    /// tile (the first versions, kept for comparison: large N then varied by 30-100 % between
    /// runs, medium N paid the pipeline fill and the store of every tile).
    /// automatic (the default): persistent when the whole batch has at least as many tiles as the
    /// GPU has SMs, else one tile per block (H100 batches of 49 - 2401 leaves of 512 - 2048:
    /// persistent 7 - 17 % faster; see wgmma_gemm_t::gemm() in src/wgmma_gemm.cu).
    enum class schedule_t { automatic, persistent, one_tile_per_block };
    static constexpr schedule_t DEFAULT_SCHEDULE = schedule_t::automatic;
    void set_schedule(schedule_t schedule) { m_schedule = schedule; }
    schedule_t schedule() const { return m_schedule; }

    /// CTAs per cluster: 1, or 2 = two vertically adjacent tiles (same tile column) per cluster,
    /// whose CTAs share B: each loads half of B's boxes with TMA multicast into both, so B is read
    /// once per pair (cuBLAS uses clusters of 2 - 16 CTAs on the H100 for every N).
    /// Default 1: on the H100 (gemm-compare, automatic g) pairs did not pay off overall yet -
    /// persistent 16384 -5.4 % vs -4.7 % (c1), but 32768 -17.7 % vs -21.2 %, 65536 +27 % vs ~0 %.
    static constexpr int DEFAULT_CLUSTER = 1;
    void set_cluster(int ctas) { m_cluster = ctas; }
    int cluster() const { return m_cluster; }

    /// Round barrier for the persistent schedule (cluster 1, static stride): every CTA's producer
    /// waits before each new round until all CTAs have reached it, so the rounds stay in step and
    /// keep sharing their k slices in L2 (src/wgmma_gemm.cu). Off by default: an experiment for
    /// very long tiles (N = 65536: 1024 k slices per tile, ~1000 rounds, times varying by ~11 %).
    static constexpr bool DEFAULT_ROUND_SYNC = false;
    void set_round_sync(bool sync) { m_round_sync = sync; }
    bool round_sync() const { return m_round_sync; }
    /// The round barrier is also used automatically from this many k slices per tile on (0: never
    /// automatically). H100: N = 65536 (1024 slices) 651 - 656 ms with it against 750 - 780 without,
    /// 32768 (512) neutral, short tiles slower (2048: +5 %); so the default 512 (N / leaf >= 32768).
    static constexpr int DEFAULT_ROUND_SYNC_THRESHOLD = 512;
    void set_round_sync_threshold(int k_tiles) { m_round_sync_threshold = k_tiles; }
    /// ... and from this much work per CTA on: k slices per tile x rounds (tiles per persistent
    /// CTA), 0: never. Without the barrier the CTAs drift apart over many rounds and stop sharing
    /// their A / B panels in L2. H100, test_diagnostics batched-compare: 49 x 16384 (256 slices,
    /// ~3040 rounds: 0.78 M) 493 ms with it against 589 without (cuBLAS 526); 7 x 16384 (0.11 M)
    /// and 49 x 8192 (0.10 M) neutral; short tiles (343 x 4096, 2401 x 2048: < 0.1 M), where a
    /// barrier per round costs the most, stay below; so 2^18.
    static constexpr long long DEFAULT_ROUND_SYNC_WORK = 1LL << 18;
    void set_round_sync_work(long long k_tiles_x_rounds) { m_round_sync_work = k_tiles_x_rounds; }

    /// Tile width (src/wgmma_gemm.cu): BN = 256, or 128 (128 x 128 tiles: wgmma m64n128k16, 6 stages,
    /// twice the CTAs with half the work each) for batches that would leave SMs idle (N = 1024: 32
    /// tiles of 128 x 256 on 132 SMs, 64 of 128 x 128). AUTO_TILE_N (0, the default): 128 when all
    /// 128-wide tiles fit on the SMs at once (H100: plain N <= 1024; at 2048 they would need two
    /// waves and were 10 % slower), with cluster 1, else 256. 128 needs cluster 1.
    static constexpr int AUTO_TILE_N = 0;
    static constexpr int DEFAULT_TILE_N = AUTO_TILE_N;
    void set_tile_n(int columns) { m_tile_n = columns; }
    int tile_n() const { return m_tile_n; }

    /// Store of C (src/wgmma_gemm.cu, epilogue): true (the default) = staged, each consumer warp
    /// writes its rows into its own shared memory buffer and sends them to global memory as whole
    /// 512 B rows with asynchronous bulk copies, so the warps return to the tensor cores sooner and
    /// every global write is fully used (needs ldc % 8 == 0 and 16 B aligned products, else direct).
    /// false = direct half2 stores from the registers (the first version, for comparisons).
    static constexpr bool DEFAULT_STAGED_STORE = true;
    void set_staged_store(bool staged) { m_staged_store = staged; }

    /// Upper bound of the persistent grid in CTAs (0, the default: every block / cluster that fits at
    /// once), to leave SMs to kernels that run concurrently (DFSCaps's overlapped preparation)
    void set_persistent_blocks(int blocks) { m_persistent_blocks = blocks; }

    /// The launch parameters of the last gemm() as text (grid, schedule, resident blocks / clusters,
    /// cluster, group, units, k tiles, store, tile), for reports such as test_diagnostics
    /// gemm-compare; made on request from the values gemm() keeps (not on every call)
    std::string last_launch() const;

    /// True when this process can run the kernel: current device of compute capability 9.0 and
    /// sm_90a code compiled in. Never throws. @p why receives the reason when false.
    static bool available(std::string &why);
    /// Whether M x M products are supported (M > 0, multiple of BN)
    static bool supports(int dimension) { return dimension > 0 && dimension % BN == 0; }
    /// Description for reports
    static std::string config_name();

    /**
     * @brief Encodes the tensor maps of a batch and uploads them (synchronous: not inside a
     *        graph capture)
     * @param A, B      host arrays of `count` device pointers (entry i: operands of product i)
     * @param lda, ldb  leading dimensions in halfs (multiples of 8)
     * @param dimension M (multiple of BN)
     * @param count     number of products, 1 .. 65535 (blockIdx.z)
     * @param key       any pointer identifying the batch (stored in plan_t::key, see gemm())
     */
    static plan_t make_plan(const half *const *A,
                            int lda,
                            const half *const *B,
                            int ldb,
                            int dimension,
                            int count,
                            const void *key);

    /// make_plan() over device pointer arrays (copied to the host first; inline: thrust ABI)
    static plan_t make_plan(const thrust::device_vector<half *> &A,
                            int lda,
                            const thrust::device_vector<half *> &B,
                            int ldb,
                            int dimension)
    {
        CV_Assert(B.size() == A.size());
        const thrust::host_vector<half *> HA = A;
        const thrust::host_vector<half *> HB = B;
        return make_plan(HA.data(), lda, HB.data(), ldb, dimension, static_cast<int>(HA.size()),
                         thrust::raw_pointer_cast(A.data()));
    }

    /**
     * @brief Batched C[i] = A_plan[i] * B_plan[i]
     * @param plan  from make_plan() (its count and dimension are used)
     * @param C     device array of plan.count device pointers
     * @param ldc   leading dimension of C in halfs (even)
     * @param accumulator half (wgmma .f16 accumulators, stored as is) or single (.f32, rounded once)
     * Asynchronous on the stream of set_stream(); only a kernel launch, so it can be captured.
     */
    void gemm(const plan_t &plan,
              half *const *C,
              int ldc,
              accumulator_t accumulator);

    /// Most targets per product of gemm_reduce() (2^R for the leaves of R levels, R <= 4)
    static constexpr int MAX_REDUCE_TARGETS = 16;

    /**
     * @brief Batched C_t[i] += ±A_plan[i] * B_plan[i] for the targets t of every product
     * @details The leaf GEMM of CAPS without product buffers: every product is added straight into the
     * blocks of the result it contributes to (the decoding matrix of the alternative basis), by bulk
     * reduce-add in the epilogue (cp.reduce.async.bulk .add.noftz.f16, element-wise atomic, so
     * products that add into the same block may run at the same time; the order of the additions,
     * hence the rounding, is not fixed).
     * @param targets  device array of plan.count * per_product pointers: product i adds into
     *                 targets[i * per_product + t], t = 0 .. per_product - 1; nullptr ends a product's
     *                 list; bit 0 set = subtract. 16 B aligned (ldc % 8 == 0), staged store on.
     * @param per_product 1 .. MAX_REDUCE_TARGETS
     */
    void gemm_reduce(const plan_t &plan, half *const *targets, int per_product, int ldc, accumulator_t accumulator)
    {
        CV_Assert(per_product >= 1 && per_product <= MAX_REDUCE_TARGETS && ldc % 8 == 0);
        m_reduce_targets = per_product;
        try {
            gemm(plan, targets, ldc, accumulator);
        } catch (...) {
            m_reduce_targets = 0;
            throw;
        }
        m_reduce_targets = 0;
    }

    /// The same with a thrust pointer array for C; checks that @p A is the array the plan was
    /// made from (plan.key), which catches a plan left over from an earlier initialize()
    void gemm(const plan_t &plan,
              const thrust::device_vector<half *> &A,
              thrust::device_vector<half *> &C,
              int ldc,
              accumulator_t accumulator)
    {
        CV_Assert(plan.valid() && plan.key == thrust::raw_pointer_cast(A.data()));
        CV_Assert(static_cast<int>(C.size()) == plan.count);
        gemm(plan, thrust::raw_pointer_cast(C.data()), ldc, accumulator);
    }

private:
    cudaStream_t m_stream = nullptr;
    int m_order = DEFAULT_ORDER; ///< set_order()
    schedule_t m_schedule = DEFAULT_SCHEDULE; ///< set_schedule()
    int m_cluster = DEFAULT_CLUSTER; ///< set_cluster()
    /// last_launch(): the parameters of the last launch (blocks 0: none yet)
    struct launch_info_t
    {
        int blocks = 0;
        bool persistent = false;
        int resident = 0;
        int cluster = 0;
        bool attribute = false;
        int group = 0;
        int units = 0;
        int k_tiles = 0;
        bool round_sync = false;
        bool staged = false;
        int reduce_targets = 0;
        int tile_n = 0;
    };
    launch_info_t m_last;
    int m_reduce_targets = 0; ///< gemm_reduce() in progress: targets per product, else 0
    bool m_round_sync = DEFAULT_ROUND_SYNC; ///< set_round_sync()
    int m_round_sync_threshold = DEFAULT_ROUND_SYNC_THRESHOLD; ///< set_round_sync_threshold()
    long long m_round_sync_work = DEFAULT_ROUND_SYNC_WORK;     ///< set_round_sync_work()
    bool m_staged_store = DEFAULT_STAGED_STORE; ///< set_staged_store()
    int m_persistent_blocks = 0;                ///< set_persistent_blocks()
    int m_tile_n = DEFAULT_TILE_N;              ///< set_tile_n()
};
