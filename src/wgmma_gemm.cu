/**
 * @file wgmma_gemm.cu
 * @brief Batched HGEMM for Hopper: TMA + mbarrier pipeline + wgmma.mma_async (include/wgmma_gemm.hpp)
 *
 * The batch is cut into BM x TN tiles, TN = BN = 256 or, for batches with fewer 256-wide tiles
 * than SMs, TN = 128 (wgmma_gemm_t::set_tile_n(); kernel template parameter TN: wgmma m64n128k16,
 * 6 stages of 32 KB instead of 4 of 48 KB, the same tensor maps: B is loaded as 2 boxes of 64
 * columns instead of 4), with one linear tile index over all products:
 * product z = tile / tiles per product, and within a product the grouped order, groups of `group`
 * tile rows, column by column within a group (wgmma_gemm_t::set_order(): L2 reuse of A row panels
 * and B column panels; group 1 = row by row). Block b (one CTA, no clusters yet) computes the tiles
 * b, b + gridDim.x, ... (tile_of()), with one of two schedules (wgmma_gemm_t::set_schedule(); the
 * default, automatic, takes persistent when one product has at least as many tiles as there are
 * SMs, else one tile per block):
 *   persistent: one block per SM. The producer prefetches the tensor maps of its next tile's
 *       product while the current tile runs. Static stride: all blocks start a round together and,
 *       the tiles being equal, stay in step, so they read the same k slices of the same A and B
 *       panels at about the same time and share them in L2. (A dynamic scheduler, tiles from a global
 *       counter, was 2x slower for long tiles on the H100 - the blocks drift apart and lose the L2
 *       sharing - and was removed.) The
 *       producer tells the consumers each tile's index through a 2-entry shared memory slot,
 *       written before the arrival on the tile's first full barrier, and ends them with an "end"
 *       entry (-1) and a plain arrival (both schedules). The SMs always work on gridDim.x consecutive tiles,
 *       so the L2 working set is the same in every round, and a block's producer loads the next
 *       tile while its consumers store the current one. (With one tile per block, blocks finish
 *       at different times, the set of tiles running together drifts apart and large N ran
 *       30-100 % slower or not, depending on the run; medium N paid the pipeline fill and the
 *       store of every tile without overlap.)
 *   one tile per block: gridDim.x = number of tiles (the first versions).
 *
 * Warp specialization, 3 warpgroups of 128 threads (384 threads, 1 block per SM):
 *   warpgroup 0, the producer: one thread issues the TMA copies of the k slices into a ring of
 *       STAGES shared memory stages. It needs almost no registers, so the warpgroup gives them
 *       back (setmaxnreg.dec 40).
 *   warpgroups 1 and 2, the consumers: each owns 64 rows of the tile (64 x 256 accumulators =
 *       128 f32 or 64 packed f16x2 registers per thread) and issues one wgmma m64n256k16 per
 *       16 wide k step, reading A and B directly from shared memory. They take the registers the
 *       producer released (setmaxnreg.inc 232): 128 x 40 + 256 x 232 = 64512 <= 65536.
 *
 * Pipeline (mbarriers in shared memory, two per stage):
 *   full[s]  (arrival count 1): the producer arrives with expect_tx(STAGE_BYTES); TMA completes
 *            the transaction bytes, so the phase flips when all of stage s has landed.
 *   empty[s] (arrival count 8): every consumer warp arrives once it has finished reading stage s,
 *            so the producer may overwrite it. One wgmma group stays in flight: after issuing
 *            slice kt, a consumer waits for the group of slice kt - 1 (wgmma.wait_group 1) and
 *            releases that slice's stage.
 *   Slice kt uses stage s = kt % STAGES in round r = kt / STAGES. Consumers wait full[s] for
 *   parity r & 1; the producer waits empty[s] for parity (r & 1) ^ 1, which for r = 0 refers to
 *   the phase "before" the first one and therefore passes at once: the ring starts empty.
 *
 * Shared memory per stage (48 KB, 1024 B aligned as the 128 B swizzle requires):
 *   A slice: 128 rows x 64 halfs, one TMA box {64, 128}; each row is one 128 B swizzle row.
 *            K-major (k contiguous) for wgmma.
 *   B slice: 64 k rows x 256 columns as 4 TMA boxes {64 columns, 64 rows} of 8 KB (the 128 B
 *            swizzle limits a box row to 128 B = 64 halfs). N-major ("transposed", tnspB = 1).
 *
 * wgmma matrix descriptors (64 bit, PTX ISA "Matrix Descriptor Format"):
 *   bits  0-13 start address >> 4, bits 16-29 leading byte offset (LBO) >> 4,
 *   bits 32-45 stride byte offset (SBO) >> 4, bits 49-51 base offset (0: atoms 1024 B aligned),
 *   bits 62-63 swizzle mode (1 = 128 B).
 *   The 128 B swizzle atom is 8 rows x 128 B = 1024 B.
 *   A, K-major: SBO = 1024 (next group of 8 rows); LBO is unused for swizzled K-major layouts
 *       (set to 16 like CUTLASS). k step 16 halfs = +32 B on the start address (the hardware
 *       applies the swizzle to the full address, so advancing inside the atom is legal; CUTLASS
 *       and DeepGEMM do the same). Consumer c starts at row 64 c = +8192 B.
 *   B, N-major: within an atom 64 columns are contiguous and the 8 rows are 8 k values.
 *       SBO = 1024 (next 8 k rows), LBO = 8192 (next 64 columns = next TMA box).
 *       k step 16 = two atoms = +2048 B.
 *
 * Epilogue: the wgmma D fragment of m64nNk16 has the layout of mma.sync's C fragment repeated
 * along N: warp w of the warpgroup owns rows 16 w .. 16 w + 15; f32 register i holds
 *   row 16 w + lane / 4 + 8 * ((i / 2) % 2), column 8 * (i / 4) + 2 * (lane % 4) + i % 2,
 * and the f16 variant packs the pairs (i, i + 1) of that list into one register j = i / 2.
 * Staged store (wgmma_gemm_t::set_staged_store(), the default): each consumer warp owns 16 rows x
 * 256 columns of the tile and stores them in two halves of 8 rows (register pairs with
 * (i / 2) % 2 == h, f16: j % 2 == h). A half is written as half2 into the warp's own 8 x 256
 * buffer in shared memory (rows padded to 528 B: the 8 rows x 4 lanes of a store hit 32 different
 * banks), then lanes 0 - 7 each send one 512 B row to global memory with a bulk copy
 * (cp.async.bulk.global.shared::cta, 1D: no tensor map, so C may change from call to call). The
 * warp goes straight back to the next tile; before it writes the buffer again each lane waits
 * until the shared memory of its own copy has been read (cp.async.bulk.wait_group.read). Every
 * global write is a whole 512 B row, against 8 rows x 16 B (half filled 32 B sectors) per
 * instruction of the direct store. Needs ldc % 8 == 0 and 16 B aligned products (checked: the
 * host checks ldc, the kernel each product's address), else the direct store below.
 * Direct store (set_staged_store(false), the first version): half2 straight from the registers.
 *
 * Clusters (wgmma_gemm_t::set_cluster(2); the default is 1, see wgmma_gemm_t::DEFAULT_CLUSTER): the
 * work unit is then a pair of tiles stacked
 * in one tile column, computed by the two CTAs of a cluster (rank r: the r-th tile row of the
 * pair). Both need the same B slice of every stage, so each CTA loads its own A box and every
 * second B box (j % 2 == rank) with TMA multicast into both CTAs: B is read once per pair, and
 * each producer issues 3 instead of 5 copies per stage. A stage's full barrier still expects 48 KB
 * (the peer's boxes complete on it too). Its empty barrier counts the consumer warps of both CTAs
 * (16 arrivals: each warp arrives locally and, through mapa, on the peer), because the peer's
 * multicast writes into this CTA's stage. A cluster barrier after the barrier init and one before
 * exit keep every CTA's shared memory and barriers alive while the peer may still use them. The
 * persistent grid is the number of clusters that fit at once (cudaOccupancyMaxActiveClusters).
 * Only real clusters are launched with the cluster attribute: with a 1 x 1 x 1 cluster attribute
 * the 132-block persistent grid ran 2.3x slower (part of the blocks only started when others had
 * finished), one block per tile was not affected (WGMMA_CLUSTER_ATTR_FOR_1=1 reproduces it).
 *
 * Not done yet: a TMA (tensor) store epilogue, clusters of 4+ / A multicast, two consumer warpgroups on
 * different tiles (ping-pong) so that one stores while the other computes.
 *
 * Arch guard: everything that uses sm_90a instructions is compiled only when
 * __CUDA_ARCH_FEAT_SM90_ALL is defined (nvcc arch 90a). In other device passes (e.g. sm_89 of a
 * "89;90a-real" build) the kernel body is empty; available() detects that with a probe kernel,
 * so gemm() never launches an empty kernel.
 */

#include "wgmma_gemm.hpp"

#include <cuda.h> // CUtensorMap and its enums (header only: the encoder comes from the driver entry point)
#include <cuda_runtime.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

/// The register handover of the warpgroups (setmaxnreg: producer 40, consumers 232). CMake option
/// WGMMA_SETMAXNREG (default ON); OFF builds the kernel without it, every region then within the
/// launch's 168 registers (a test for the batched benchmark's crash, which started with the first
/// non-rdc build, where setmaxnreg took effect). A compile-time switch: skipping the instructions
/// at run time would leave the consumer code using registers it does not own.
#ifndef WGMMA_SETMAXNREG
#define WGMMA_SETMAXNREG 1
#endif
/// prefetch.tensormap of the tensor maps (the current tile's product and, one unit ahead, the
/// next one's). CMake option WGMMA_PREFETCH_TENSORMAP (default ON); OFF: no prefetch, the first
/// TMA copy of a product fetches its descriptor (a test for the batched benchmark's crash).
#ifndef WGMMA_PREFETCH_TENSORMAP
#define WGMMA_PREFETCH_TENSORMAP 1
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define WGMMA_DEVICE_CODE 1 ///< this device pass may use wgmma / TMA / setmaxnreg
#else
#define WGMMA_DEVICE_CODE 0
#endif

namespace {

constexpr int BM = wgmma_gemm_t::BM;
constexpr int BN = wgmma_gemm_t::BN;
constexpr int BK = wgmma_gemm_t::BK;
constexpr int STAGES = wgmma_gemm_t::STAGES;

constexpr int CONSUMERS = 2;                        ///< consumer warpgroups (64 rows each)
constexpr int THREADS = 128 * (1 + CONSUMERS);      ///< producer warpgroup + consumers
constexpr int WGMMA_M = 64;                         ///< rows of one wgmma (fixed by the ISA)
constexpr int A_TILE_BYTES = BM * BK * 2;           ///< 16 KB
constexpr int B_BOX_COLS = 64;                      ///< columns of one B box: 128 B rows
constexpr int B_TILE_BYTES = BN * BK * 2;           ///< 32 KB
constexpr int STAGE_BYTES = A_TILE_BYTES + B_TILE_BYTES; ///< 48 KB, also the expect_tx count
constexpr int SWIZZLE_ATOM_BYTES = 1024;            ///< 8 rows x 128 B
constexpr int EPI_ROWS = 8;                         ///< rows of one staged store half (per warp)
constexpr int EPI_ROW_BYTES = BN * 2 + 16;          ///< 512 B of a C row + 16 B padding (banks)
constexpr int EPI_WARP_BYTES = EPI_ROWS * EPI_ROW_BYTES; ///< 4224 B: one consumer warp's buffer
constexpr int EPI_BYTES = CONSUMERS * 4 * EPI_WARP_BYTES; ///< 33 KB: the 8 consumer warps
/// stages + store buffers + slack for the alignment
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + EPI_BYTES + SWIZZLE_ATOM_BYTES;
constexpr int PRODUCER_REGS = 40;                   ///< setmaxnreg.dec (multiple of 8)
constexpr int CONSUMER_REGS = 232;                  ///< setmaxnreg.inc (multiple of 8)

static_assert(BM == WGMMA_M * CONSUMERS, "each consumer warpgroup owns 64 rows");
static_assert(BK * 2 == 128, "a k slice row must be exactly one 128 B swizzle row");
static_assert(BN % B_BOX_COLS == 0, "B tile = whole boxes");
static_assert(BN == 256, "the wgmma wrappers below are m64n256k16 (and m64n128k16 for TN = 128)");
constexpr int TN_SMALL = 128;                       ///< the narrow tile width (set_tile_n())
constexpr int STAGES_SMALL = 6;                     ///< stages of the narrow tile: 6 x 32 KB

/// Stages and bytes per stage of a tile width (TN = BN: 4 x 48 KB, TN = 128: 6 x 32 KB; both 192 KB,
/// so the store buffers start at the same offset)
/// (__host__ __device__: also used by the kernel; nvcc rejects host-only constexpr calls there)
__host__ __device__ constexpr int stages_of(int tn) { return tn == BN ? STAGES : STAGES_SMALL; }
__host__ __device__ constexpr int stage_bytes_of(int tn) { return A_TILE_BYTES + tn * BK * 2; }
static_assert(stages_of(TN_SMALL) * stage_bytes_of(TN_SMALL) == STAGES * STAGE_BYTES, "same stage area");
static_assert(A_TILE_BYTES % SWIZZLE_ATOM_BYTES == 0 && STAGE_BYTES % SWIZZLE_ATOM_BYTES == 0,
              "every tile and box must start on a swizzle atom");
static_assert(128 * PRODUCER_REGS + 128 * CONSUMERS * CONSUMER_REGS <= 65536, "register file");
static_assert(EPI_ROW_BYTES % 16 == 0 && EPI_WARP_BYTES % 16 == 0, "bulk copies need 16 B aligned rows");
// 227 KB per block on the H100 (dynamic + static: the barriers and unit slots, < 128 B)
static_assert(SMEM_BYTES + 128 <= 227 * 1024, "shared memory per block");

#if WGMMA_DEVICE_CODE

// Constants used only by the sm_90a kernel body (outside this block they would be unused in the
// other device passes, e.g. sm_89: nvcc warning #177)
constexpr int WGMMA_K = 16;                         ///< k of one wgmma for f16 inputs
constexpr int B_BOX_BYTES = B_BOX_COLS * BK * 2;    ///< 8 KB, one B box
constexpr int EMPTY_ARRIVALS = CONSUMERS * 4;       ///< one arrival per consumer warp

/// Shared-space (32 bit) address of a shared memory pointer, as the PTX instructions want it
__device__ __forceinline__ uint32_t smem_u32(const void *p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// mbarrier.init: expected arrivals per phase
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

/// One arrival plus `bytes` expected transaction bytes (completed by the TMA copies)
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
                 : "memory");
}

/// One plain arrival
__device__ __forceinline__ void mbar_arrive(uint64_t *bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

/// Spins until the phase with parity `parity` of the barrier has completed
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity)
{
    uint32_t done = 0;
    do {
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "  selp.u32 %0, 1, 0, p;\n"
                     "}\n"
                     : "=r"(done)
                     : "r"(smem_u32(bar)), "r"(parity)
                     : "memory");
    } while (!done);
}

/// TMA: box at coordinates (c0 = innermost / column, c1 = row) of `map` -> dst; completes bytes
/// on `bar`. The tensor map lives in global memory (written by the host before the launch).
__device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *map, uint64_t *bar, int c0, int c1)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
                 : "memory");
}

/// Acquire fence of the tensor-map proxy for one map in global memory: the host wrote the maps
/// (make_plan(): cudaMemcpy), and plans are made and freed again and again, so a later plan's
/// maps can sit at the address of an earlier one's. Without this fence the TMA unit may use a
/// stale descriptor it cached from the earlier map at that address (PTX: a tensor map modified
/// in global memory must be fenced with fence.proxy.tensormap::generic.acquire before its use;
/// CUTLASS does the same for descriptors in global memory). Seen as: the batched benchmark (plans
/// of 343 leaves after many freed smaller plans) died with illegal instructions / addresses, in
/// our kernel or the next one, while the same launch alone ran 20000 times cleanly.
__device__ __forceinline__ void fence_tensormap(const CUtensorMap *map)
{
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(reinterpret_cast<uint64_t>(map))
                 : "memory");
}

/// Brings a tensor map into the TMA unit's cache ahead of the first copy
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap *map)
{
#if WGMMA_PREFETCH_TENSORMAP
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
#else
    (void) map;
#endif
}

/// Orders this thread's earlier generic shared memory writes before later async-proxy accesses
/// (the bulk copies read the staged C rows through the async proxy)
__device__ __forceinline__ void fence_proxy_async_shared()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

/// 1D bulk copy shared -> global of `bytes` (multiple of 16, both addresses 16 B aligned), in this
/// thread's current bulk group
__device__ __forceinline__ void bulk_store(void *dst, const void *src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(reinterpret_cast<uint64_t>(dst)),
                 "r"(smem_u32(src)), "r"(bytes)
                 : "memory");
}

/// 1D bulk reduce-add shared -> global of `bytes` halves' bytes (C[i] += S[i], f16, element-wise atomic
/// against other reductions; same alignment rules as bulk_store())
__device__ __forceinline__ void bulk_reduce_add(void *dst, const void *src, uint32_t bytes)
{
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
                 ::"l"(reinterpret_cast<uint64_t>(dst)), "r"(smem_u32(src)), "r"(bytes)
                 : "memory");
}

/// Closes this thread's current bulk group
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

/// Waits until at most N of this thread's bulk groups still read their shared memory source
template <int N>
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }

/// Waits until at most N of this thread's bulk groups are incomplete (writes done)
template <int N>
__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory"); }

/// Rank of this CTA in its cluster (0 .. cluster size - 1)
__device__ __forceinline__ uint32_t cluster_ctarank()
{
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

/// Barrier over all threads of all CTAs of the cluster (every thread must execute it)
__device__ __forceinline__ void cluster_sync()
{
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

/// One arrival on the mbarrier at the same shared memory offset in the peer CTA `cta` of the
/// cluster (mapa: local shared address -> the peer's shared::cluster address). The plain form, as
/// CUTLASS's ClusterBarrier::arrive(cta_id): the first version used
/// mbarrier.arrive.release.cluster, a cluster-wide memory fence on every arrival (two per warp per
/// k slice), and the cluster kernel ran at 40 % of the speed.
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t *bar, uint32_t cta)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(bar)), "r"(cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote) : "memory");
}

/// TMA with multicast: the box lands at the same shared memory offset in every CTA of `mask`, and
/// its bytes complete on the mbarrier at `bar`'s offset in each of them
__device__ __forceinline__ void tma_load_2d_multicast(void *dst, const CUtensorMap *map, uint64_t *bar, int c0, int c1,
                                                      uint16_t mask)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_u32(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar)), "h"(mask)
                 : "memory");
}

/// Matrix descriptor for a 128 B swizzled operand (see the file comment)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo)
{
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4);
    desc |= static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16;
    desc |= static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32;
    desc |= 1ull << 62; // swizzle mode 1: 128 B
    return desc;
}

/// Orders register accesses to the accumulators before / after wgmma: wgmma reads and writes
/// them asynchronously, so the compiler must not move other uses across fence / wait
__device__ __forceinline__ void fence_operand(float &r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_operand(uint32_t &r) { asm volatile("" : "+r"(r)::"memory"); }

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

/// setmaxnreg: change this warpgroup's register budget (all its warps must execute it)
template <int REGS>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(REGS)); }
template <int REGS>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(REGS)); }

/// D += A * B, m64n256k16, f32 accumulators (128 per thread). scale-d = 1 (accumulate; the
/// registers start at zero), A K-major (tnspA 0), B N-major (tnspB 1), no negation.
__device__ __forceinline__ void wgmma_m64n256k16(float (&d)[128], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %130, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "  {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n"
        "   %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,\n"
        "   %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,\n"
        "   %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,\n"
        "   %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,\n"
        "   %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,\n"
        "   %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,\n"
        "   %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "  %128, %129, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}

/// The same with f16 accumulators: 64 registers, each a packed pair of halfs
__device__ __forceinline__ void wgmma_m64n256k16(uint32_t (&d)[64], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %66, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16\n"
        "  {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n"
        "   %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,\n"
        "   %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,\n"
        "   %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "  %64, %65, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}

/// D += A * B, m64n128k16 (TN = 128 tiles), f32 accumulators: 64 per thread
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %66, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "  {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n"
        "   %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,\n"
        "   %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,\n"
        "   %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "  %64, %65, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}

/// m64n128k16 with f16 accumulators: 32 registers of packed halfs
__device__ __forceinline__ void wgmma_m64n128k16(uint32_t (&d)[32], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %34, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16\n"
        "  {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,\n"
        "   %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "  %32, %33, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(1));
}

/// The wgmma of a tile width: the accumulator array decides (n256: 128 f32 / 64 f16 registers,
/// n128: 64 / 32)
__device__ __forceinline__ void wgmma_tile(float (&d)[128], uint64_t a, uint64_t b) { wgmma_m64n256k16(d, a, b); }
__device__ __forceinline__ void wgmma_tile(uint32_t (&d)[64], uint64_t a, uint64_t b) { wgmma_m64n256k16(d, a, b); }
__device__ __forceinline__ void wgmma_tile(float (&d)[64], uint64_t a, uint64_t b) { wgmma_m64n128k16(d, a, b); }
__device__ __forceinline__ void wgmma_tile(uint32_t (&d)[32], uint64_t a, uint64_t b) { wgmma_m64n128k16(d, a, b); }

#endif // WGMMA_DEVICE_CODE

/// Accumulator registers of one consumer thread: 64 x TN per warpgroup / 128 threads (f32: TN / 2
/// values, f16: TN / 4 packed pairs)
template <bool F32_ACC, int TN>
struct accumulators_t;
template <int TN>
struct accumulators_t<true, TN>
{
    static constexpr int COUNT = TN / 2;
    float r[COUNT];
};
template <int TN>
struct accumulators_t<false, TN>
{
    static constexpr int COUNT = TN / 4;
    uint32_t r[COUNT];
};

/// One tile of the batch: product z, first row m0 and first column n0 of its BM x BN block of C
struct tile_t
{
    int z;
    int m0;
    int n0;
};

#if WGMMA_DEVICE_CODE
/**
 * @brief Linear tile index -> tile: product z = tile / (tiles_m * tiles_n), within the product the
 *        grouped order (groups of `group` tile rows, column by column; the last group may have
 *        fewer rows). Consecutive indices are the tiles that should run at the same time.
 */
__device__ __forceinline__ tile_t tile_of(int tile, int tiles_m, int tiles_n, int group, int tile_n)
{
    const int per_product = tiles_m * tiles_n;
    const int z = tile / per_product;
    const int i = tile % per_product;
    const int per_group = group * tiles_n;
    const int first_row = (i / per_group) * group;
    const int rows = min(tiles_m - first_row, group);
    const int in_group = i % per_group;
    tile_t result;
    result.z = z;
    result.m0 = (first_row + in_group % rows) * BM;
    result.n0 = (in_group / rows) * tile_n;
    return result;
}
#endif

/**
 * @brief The batched GEMM kernel (see the file comment)
 * @details Work unit = CLUSTER tiles stacked in one tile column (CLUSTER consecutive tile rows, the
 * same columns): a cluster computes units u, u + clusters, ... < total_units, its CTA of rank r
 * the r-th tile of each unit. The units are numbered like tiles (tile_of() over tiles_m / CLUSTER
 * unit rows), so the grouped order and both schedules (persistent / one unit per cluster) work as
 * for single CTAs. With CLUSTER = 2 both CTAs need the same B slice: each loads its own A and half
 * of B's boxes with multicast into both, so B is read once per pair.
 * Producer and consumers count the k slices over all units of the CTA (it): stage it % STAGES,
 * barrier phase (it / STAGES) & 1. A stage is free again only when the consumers of every CTA of
 * the cluster have read it (the peers write into it by multicast), so every consumer warp arrives
 * on the empty barrier of each CTA: EMPTY_ARRIVALS * CLUSTER arrivals per phase.
 * @tparam CLUSTER 1 (no cluster) or 2 (pairs of CTAs, launched with the cluster attribute)
 * @tparam TN      tile columns: BN (256) or TN_SMALL (128, with CLUSTER = 1)
 * @param maps        2 * count tensor maps: A[z] at 2z, B[z] at 2z + 1
 * @param C           count device pointers to the products
 * @param ldc         leading dimension of C in halfs
 * @param k_tiles     M / BK
 * @param tiles_m     tile rows of a product (M / BM, a multiple of CLUSTER)
 * @param tiles_n     tile columns of a product (M / BN)
 * @param group       unit rows per group of the tile order (>= 1, see the file comment)
 * @param total_units count * tiles_m * tiles_n / CLUSTER
 * @param round_counter round barrier (persistent, CLUSTER = 1, static; set_round_sync()): a device
 *                    int set to 0 before the launch. Before the first load of its unit of round s
 *                    (s >= 1) a CTA's producer adds 1 and waits until the counter reaches
 *                    s * gridDim.x, so all CTAs start every round together and keep sharing the k
 *                    slices in L2 (long tiles drifted apart over hundreds of rounds). A CTA with
 *                    fewer units (the last round is partial) adds its missing arrivals when it
 *                    ends. Needs every CTA resident at once (the persistent grid). nullptr: off.
 * @param staged      1: staged store of C through shared memory and bulk copies (needs ldc % 8 == 0,
 *                    checked by the host; products not 16 B aligned fall back per tile), 0: direct
 *                    half2 stores (see the file comment)
 */
template <bool F32_ACC, int CLUSTER, int TN>
__global__ void __launch_bounds__(THREADS, 1)
    wgmma_batched_kernel(const CUtensorMap *__restrict__ maps,
                         half *const *__restrict__ C,
                         int ldc,
                         int k_tiles,
                         int tiles_m,
                         int tiles_n,
                         int group,
                         int total_units,
                         int *__restrict__ round_counter,
                         int staged)
{
#if WGMMA_DEVICE_CODE
    static_assert(CLUSTER == 1 || CLUSTER == 2, "clusters of 1 or 2 CTAs");
    constexpr int CTAS = CLUSTER; // CTAs of the launch cluster
    static_assert(TN == BN || (TN == TN_SMALL && CTAS == 1), "128-column tiles without clusters");
    constexpr int ST = stages_of(TN);            // stages of this tile width
    constexpr int STAGE_T = stage_bytes_of(TN);  // bytes per stage (also the expect_tx count)
    constexpr int ROW_T = TN * 2 + 16;           // staged store: bytes per buffered row (padded)
    constexpr int WARP_T = EPI_ROWS * ROW_T;     // staged store: one consumer warp's buffer
    static_assert(ROW_T % 16 == 0 && WARP_T * 4 * CONSUMERS <= EPI_BYTES, "store buffers");
    __shared__ __align__(8) uint64_t full_bar[ST];
    __shared__ __align__(8) uint64_t empty_bar[ST];
    // the unit of the producer's last two units, for the consumers (-1: no more units). Two slots:
    // the producer writes unit seq + 2 only after the consumers released slices of unit seq + 1, so
    // long after they read unit seq (every unit has k_tiles >= ST slices)
    __shared__ int unit_slot[2];
    extern __shared__ uint8_t dyn_smem[];
    // align the shared-space address (what the swizzle sees) to the 1024 B atom; the same offset in
    // every CTA (same kernel, same static shared memory), which the multicast relies on
    uint8_t *smem = dyn_smem + ((SWIZZLE_ATOM_BYTES - (smem_u32(dyn_smem) % SWIZZLE_ATOM_BYTES)) % SWIZZLE_ATOM_BYTES);

    const int warpgroup = threadIdx.x / 128;
    const int t = threadIdx.x % 128; // thread in the warpgroup
    const int cta_rank = CTAS > 1 ? int(cluster_ctarank()) : 0;
    const int rank = CLUSTER > 1 ? cta_rank : 0;   // tile of the pair (CLUSTER = 2)
    const int cluster_id = blockIdx.x / CTAS;
    const int clusters = gridDim.x / CTAS;
    const int unit_rows = tiles_m / CLUSTER;
    // unit -> this CTA's tile: the unit's tile row times CLUSTER, plus the rank
    const auto my_tile = [&](int unit) {
        tile_t tt = tile_of(unit, unit_rows, tiles_n, group, TN);
        tt.m0 = ((tt.m0 / BM) * CLUSTER + rank) * BM;
        return tt;
    };

    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], EMPTY_ARRIVALS * CLUSTER);
        }
        // make the initialized barriers visible to the async proxy (TMA complete_tx) and, with
        // clusters, to the peers (their multicasts and remote arrivals)
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if constexpr (CTAS > 1) {
        cluster_sync(); // every CTA's barriers exist before any peer writes or arrives on them
    } else {
        __syncthreads();
    }

    if (warpgroup == 0) {
        // ---------------- producer ----------------
#if WGMMA_SETMAXNREG
        setmaxnreg_dec<PRODUCER_REGS>();
#endif
        if (t == 0) {
            int it = 0;            // k slices issued by this CTA so far, over all its units
            int prefetched = -1;   // product whose tensor maps were prefetched last
            int seq = 0;           // units started by this CTA so far (unit_slot[seq % 2])
            int unit = cluster_id; // the first unit is static: one per cluster
            // round barrier: rounds 1 .. rounds - 1 have one, each CTA arrives once per round
            const int rounds = (total_units + clusters - 1) / clusters;
            for (;;) {
                if (round_counter != nullptr) {
                    if (unit >= total_units) {
                        // done early (the last round is partial): arrive for the rounds left, so
                        // that nobody waits for this CTA
                        if (rounds - seq > 0) {
                            atomicAdd(round_counter, rounds - seq);
                        }
                    } else if (seq > 0) {
                        atomicAdd(round_counter, 1);
                        const int target = seq * int(gridDim.x);
                        while (*reinterpret_cast<volatile int *>(round_counter) < target) {
                            __nanosleep(64);
                        }
                    }
                }
                if (unit >= total_units) {
                    // no more work: announce an "end" unit (-1) like a unit's first slice, so the
                    // consumers wake up, read it and stop. The stage must be free first (its previous
                    // full phase consumed), else this arrival could complete a phase too early.
                    const int s = it % ST;
                    mbar_wait(&empty_bar[s], ((it / ST) & 1) ^ 1);
                    unit_slot[seq % 2] = -1;
                    mbar_arrive(&full_bar[s]); // completes the phase: no transaction bytes expected
                    break;
                }
                const tile_t tt = my_tile(unit);
                int next = total_units; // the unit after this one (fetched at its first slice)
                const CUtensorMap *map_a = maps + 2 * tt.z;
                const CUtensorMap *map_b = map_a + 1;
                if (tt.z != prefetched) {
                    // first use of this product's maps by this CTA: fence them (see fence_tensormap())
                    fence_tensormap(map_a);
                    fence_tensormap(map_b);
                    prefetch_tensormap(map_a);
                    prefetch_tensormap(map_b);
                    prefetched = tt.z;
                }
                for (int kt = 0; kt < k_tiles; ++kt, ++it) {
                    const int s = it % ST;
                    const uint32_t round_parity = (it / ST) & 1;
                    // stage free in this CTA (first round: at once); with clusters the arrivals of
                    // the peers' consumers are counted here too, so it is free in all of them
                    mbar_wait(&empty_bar[s], round_parity ^ 1);
                    uint8_t *stage = smem + s * STAGE_T;
                    if (kt == 0) {
                        // which unit this is, for the consumers: written before the arrival below,
                        // whose release orders it before their wait on this barrier
                        unit_slot[seq % 2] = unit;
                    }
                    mbar_arrive_expect_tx(&full_bar[s], STAGE_T); // A + all of B land here
                    tma_load_2d(stage, map_a, &full_bar[s], kt * BK, tt.m0); // own A rows m0.., k columns
                    uint8_t *stage_b = stage + A_TILE_BYTES;
                    if constexpr (CLUSTER == 1) {
#pragma unroll
                        for (int j = 0; j < TN / B_BOX_COLS; ++j) { // B k rows, 64 columns per box
                            tma_load_2d(stage_b + j * B_BOX_BYTES, map_b, &full_bar[s], tt.n0 + j * B_BOX_COLS, kt * BK);
                        }
                    } else {
                        // this CTA's share of B's boxes (j % CLUSTER == rank), into every CTA
#pragma unroll
                        for (int j = 0; j < TN / B_BOX_COLS; ++j) {
                            if (j % CLUSTER == rank) {
                                tma_load_2d_multicast(stage_b + j * B_BOX_BYTES, map_b, &full_bar[s],
                                                      tt.n0 + j * B_BOX_COLS, kt * BK, uint16_t((1 << CLUSTER) - 1));
                            }
                        }
                    }
                    if (kt == 0) {
                        // the next unit: every clusters-th. Taken now, one unit ahead, so that the
                        // tensor-map fetch of the next product overlaps this unit.
                        next = unit + clusters;
                        if (next < total_units) {
                            const int next_z = my_tile(next).z;
                            if (next_z != prefetched) {
                                // fenced here, ahead of their first copy (the unit after this one)
                                fence_tensormap(maps + 2 * next_z);
                                fence_tensormap(maps + 2 * next_z + 1);
                                prefetch_tensormap(maps + 2 * next_z);
                                prefetch_tensormap(maps + 2 * next_z + 1);
                                prefetched = next_z;
                            }
                        }
                    }
                }
                ++seq;
                unit = next;
            }
        }
    } else {
        // ---------------- consumers ----------------
#if WGMMA_SETMAXNREG
        setmaxnreg_inc<CONSUMER_REGS>();
#endif
        const int c = warpgroup - 1; // rows 64 c .. 64 c + 63 of the tile
        const int warp = t / 32;
        const int lane = t % 32;
        // this warp is done reading stage s: release it in every CTA of the cluster
        const auto release = [&](int s) {
            if (lane == 0) {
                mbar_arrive(&empty_bar[s]); // this CTA: a local arrival
                if constexpr (CLUSTER > 1) {
#pragma unroll
                    for (int cta = 0; cta < CLUSTER; ++cta) {
                        if (cta != rank) {
                            mbar_arrive_cluster(&empty_bar[s], uint32_t(cta)); // the peers
                        }
                    }
                }
            }
        };

        // Indexed loops over the accumulators, not range-for: nvcc did not unroll the range-for
        // (f32: a 512 B local-memory array instead of registers, and ptxas serialized the wgmma)
        using acc_t = accumulators_t<F32_ACC, TN>;
        acc_t acc;
        int it = 0; // k slices consumed by this CTA so far (the producer's count, slice by slice)

        for (int seq = 0;; ++seq) {
            // the unit's first slice: its full barrier also says which unit it is (-1: no more)
            mbar_wait(&full_bar[it % ST], (it / ST) & 1);
            const int unit = reinterpret_cast<volatile int *>(unit_slot)[seq % 2];
            if (unit < 0) {
                break;
            }
            const tile_t tt = my_tile(unit);
#pragma unroll
            for (int i = 0; i < acc_t::COUNT; ++i) {
                acc.r[i] = 0;
            }

            for (int kt = 0; kt < k_tiles; ++kt, ++it) {
                const int s = it % ST;
                if (kt > 0) {
                    mbar_wait(&full_bar[s], (it / ST) & 1); // stage s has landed (A and all of B)
                }
                const uint32_t a_base = smem_u32(smem + s * STAGE_T) + c * WGMMA_M * BK * 2;
                const uint32_t b_base = smem_u32(smem + s * STAGE_T + A_TILE_BYTES);
#pragma unroll
                for (int i = 0; i < acc_t::COUNT; ++i) {
                    fence_operand(acc.r[i]);
                }
                wgmma_fence(); // accumulators written by ordinary instructions before (zeroing / epilogue)
#pragma unroll
                for (int kk = 0; kk < BK / WGMMA_K; ++kk) {
                    const uint64_t desc_a = make_desc(a_base + kk * WGMMA_K * 2, 16, SWIZZLE_ATOM_BYTES);
                    const uint64_t desc_b = make_desc(b_base + kk * 2 * SWIZZLE_ATOM_BYTES, B_BOX_BYTES, SWIZZLE_ATOM_BYTES);
                    wgmma_tile(acc.r, desc_a, desc_b);
                }
                wgmma_commit();
                // keep this slice's group in flight: wait only for the previous one (slice it - 1),
                // so the tensor cores always have the next group queued while the warps wait /
                // release. The groups accumulate into the same registers; the hardware orders them
                // (CUTLASS's K_PIPE_MMAS = 1).
                wgmma_wait<1>();
#pragma unroll
                for (int i = 0; i < acc_t::COUNT; ++i) {
                    fence_operand(acc.r[i]);
                }
                if (kt > 0) {
                    release((it - 1) % ST); // slice it - 1 is read
                }
            }
            // the unit's last group; then release its stage too, so that the producers can already
            // load the next unit's slices while this tile is stored
            wgmma_wait<0>();
#pragma unroll
            for (int i = 0; i < acc_t::COUNT; ++i) {
                fence_operand(acc.r[i]);
            }
            release((it - 1) % ST);

            // ---------------- epilogue ----------------
            // staged: bit 0 staged store, bit 1 reduce mode (wgmma_gemm_t::gemm_reduce()): C then holds
            // targets = staged >> 8 pointers per product (C[z * targets + t], nullptr = unused; bit 0
            // set = subtract), each receiving the tile by bulk reduce-add instead of a store
            const bool reduce = (staged & 2) != 0;
            const int targets = reduce ? (staged >> 8) : 1;
            half *const *Ct = C + size_t(tt.z) * targets;
            half *Cz = reinterpret_cast<half *>(reinterpret_cast<uintptr_t>(Ct[0]) & ~uintptr_t(1));
            if ((staged & 1) != 0 && (reduce || reinterpret_cast<uintptr_t>(Cz) % 16 == 0)) { // reduce: aligned targets required
                // staged store (see the file comment): this warp's rows 16 warp + 8 h .. + 7 of its
                // consumer's 64, through its own 8 x 256 buffer behind the stages
                uint8_t *buffer = smem + ST * STAGE_T + (c * 4 + warp) * WARP_T;
                const int r = lane / 4;     // row in the half
                const int q2 = (lane % 4) * 2; // first column of the lane's pairs
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t written = 2; // sign of the buffer's contents (2: not written yet)
                    for (int t = 0; t < targets; ++t) {
                        const auto p = reinterpret_cast<uintptr_t>(Ct[t]);
                        if (p == 0) {
                            break;
                        }
                        const uint32_t negative = uint32_t(p & 1);
                        half *dst = reinterpret_cast<half *>(p & ~uintptr_t(1));
                        if (negative != written) {
                            // the buffer is free once the bulk copies of the previous half (or of the previous
                            // tile, or of the previous target) have read it: each of lanes 0 - 7 waits for its
                            // own, the others have none
                            bulk_wait_read<0>();
                            __syncwarp();
                            uint8_t *row_bytes = buffer + r * ROW_T;
                            if constexpr (F32_ACC) {
                                const float sign = negative ? -1.f : 1.f;
#pragma unroll
                                for (int i = 0; i < acc_t::COUNT; i += 4) { // pairs (i, i + 1) of row r, (i + 2, i + 3) of r + 8
                                    const int k = i + 2 * h;
                                    const int col = 8 * (i / 4) + q2;
                                    *reinterpret_cast<half2 *>(row_bytes + col * 2) =
                                        __floats2half2_rn(sign * acc.r[k], sign * acc.r[k + 1]);
                                }
                            } else {
                                const uint32_t flip = negative ? 0x80008000u : 0u; // f16 sign bits of a pair
#pragma unroll
                                for (int j = h; j < acc_t::COUNT; j += 2) { // packed pairs: even j row r, odd j row r + 8
                                    const int col = 8 * (j / 2) + q2;
                                    *reinterpret_cast<uint32_t *>(row_bytes + col * 2) = acc.r[j] ^ flip;
                                }
                            }
                            // the writes before the copies: each writer orders its own, the warp sync all
                            fence_proxy_async_shared();
                            __syncwarp();
                            written = negative;
                        }
                        if (lane < EPI_ROWS) {
                            const int row = tt.m0 + c * WGMMA_M + warp * 16 + 8 * h + lane;
                            if (reduce) {
                                bulk_reduce_add(dst + size_t(row) * ldc + tt.n0, buffer + lane * ROW_T, TN * 2);
                            } else {
                                bulk_store(dst + size_t(row) * ldc + tt.n0, buffer + lane * ROW_T, TN * 2);
                            }
                            bulk_commit();
                        }
                    }
                }
                continue;
            }
            // direct store: registers -> global, half2
            const int row0 = tt.m0 + c * WGMMA_M + warp * 16 + lane / 4;
            const int col0 = tt.n0 + (lane % 4) * 2;
            if constexpr (F32_ACC) {
#pragma unroll
                for (int i = 0; i < acc_t::COUNT; i += 2) {
                    const int row = row0 + 8 * ((i / 2) % 2);
                    const int col = col0 + 8 * (i / 4);
                    *reinterpret_cast<half2 *>(Cz + size_t(row) * ldc + col) = __floats2half2_rn(acc.r[i], acc.r[i + 1]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < acc_t::COUNT; ++j) {
                    const int row = row0 + 8 * (j % 2);
                    const int col = col0 + 8 * (j / 2);
                    *reinterpret_cast<uint32_t *>(Cz + size_t(row) * ldc + col) = acc.r[j];
                }
            }
        }
        // the last staged rows: written to global memory before this CTA ends (its shared memory
        // must also stay alive until they were read); no-op without staged copies
        bulk_wait<0>();
    }
    if constexpr (CTAS > 1) {
        // no CTA leaves while a peer may still multicast into its shared memory, arrive on its
        // barriers (all threads of all CTAs, producers included)
        cluster_sync();
    }
#endif // WGMMA_DEVICE_CODE
}

/// Opts one kernel variant in to SMEM_BYTES of dynamic shared memory; false on failure
template <bool F32_ACC, int CLUSTER, int TN>
bool reserve_shared_memory()
{
    return cudaFuncSetAttribute(wgmma_batched_kernel<F32_ACC, CLUSTER, TN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                SMEM_BYTES)
           == cudaSuccess;
}

/// Registers per thread of the register handover: the producer gives up to PRODUCER_REGS, the
/// consumers take CONSUMER_REGS, which balances only if the launch allocated exactly this many
/// (__launch_bounds__(THREADS, 1): 65536 / 384 rounded down to 8 = 168). With fewer, the consumers'
/// setmaxnreg.inc would wait for registers that never become free.
constexpr int HANDOVER_REGS = (128 * PRODUCER_REGS + 128 * CONSUMERS * CONSUMER_REGS) / THREADS;
static_assert(HANDOVER_REGS * THREADS == 128 * PRODUCER_REGS + 128 * CONSUMERS * CONSUMER_REGS,
              "the register handover must hand over whole registers per thread");

/// Prepares one kernel variant: opts in to its dynamic shared memory (225 KB: stages, store
/// buffers, slack; above the 48 KB default) and, with the register handover, checks that ptxas
/// gave it HANDOVER_REGS registers per thread. False with the reason in `why`.
template <bool F32_ACC, int CLUSTER, int TN>
bool prepare_variant(std::string &why)
{
    if (!reserve_shared_memory<F32_ACC, CLUSTER, TN>()) {
        why = "cannot reserve " + std::to_string(SMEM_BYTES) + " B of shared memory";
        return false;
    }
#if WGMMA_SETMAXNREG
    cudaFuncAttributes attributes = {};
    if (cudaFuncGetAttributes(&attributes, wgmma_batched_kernel<F32_ACC, CLUSTER, TN>) != cudaSuccess
        || attributes.numRegs != HANDOVER_REGS) {
        why = "the kernel has " + std::to_string(attributes.numRegs) + " registers per thread, the register handover ("
              + std::to_string(PRODUCER_REGS) + " / " + std::to_string(CONSUMER_REGS) + ") needs "
              + std::to_string(HANDOVER_REGS);
        return false;
    }
#endif
    return true;
}

/// Launches one kernel variant with `config` (grid, cluster attribute, stream)
template <bool F32_ACC, int CLUSTER, int TN>
cudaError_t launch_variant(const cudaLaunchConfig_t &config,
                           const CUtensorMap *maps,
                           half *const *C,
                           int ldc,
                           int k_tiles,
                           int tiles_m,
                           int tiles_n,
                           int group,
                           int total_units,
                           int *rounds,
                           int staged)
{
    return cudaLaunchKernelEx(&config, wgmma_batched_kernel<F32_ACC, CLUSTER, TN>, maps, C, ldc, k_tiles, tiles_m,
                              tiles_n, group, total_units, rounds, staged);
}

/// The variant for (cluster, tile width): 128-column tiles (1, 128), pairs (2, 256), else (1, 256)
template <bool F32_ACC>
cudaError_t launch_for(int cluster,
                       int tile_n,
                       const cudaLaunchConfig_t &config,
                       const CUtensorMap *maps,
                       half *const *C,
                       int ldc,
                       int k_tiles,
                       int tiles_m,
                       int tiles_n,
                       int group,
                       int total_units,
                       int *rounds,
                       int staged)
{
    if (tile_n == TN_SMALL) {
        return launch_variant<F32_ACC, 1, TN_SMALL>(config, maps, C, ldc, k_tiles, tiles_m, tiles_n, group, total_units, rounds,
                                                    staged);
    }
    if (cluster == 2) {
        return launch_variant<F32_ACC, 2, BN>(config, maps, C, ldc, k_tiles, tiles_m, tiles_n, group, total_units, rounds, staged);
    }
    return launch_variant<F32_ACC, 1, BN>(config, maps, C, ldc, k_tiles, tiles_m, tiles_n, group, total_units, rounds, staged);
}

/// Writes 1 when this device pass contains the real kernel (sm_90a), 0 otherwise
__global__ void probe_kernel(int *out)
{
    *out = WGMMA_DEVICE_CODE;
}

/// cuTensorMapEncodeTiled (driver API), looked up through the runtime: no libcuda link needed
using encode_tiled_fn_t = CUresult (*)(CUtensorMap *,
                                       CUtensorMapDataType,
                                       cuuint32_t,
                                       void *,
                                       const cuuint64_t *,
                                       const cuuint64_t *,
                                       const cuuint32_t *,
                                       const cuuint32_t *,
                                       CUtensorMapInterleave,
                                       CUtensorMapSwizzle,
                                       CUtensorMapL2promotion,
                                       CUtensorMapFloatOOBfill);

encode_tiled_fn_t encode_tiled()
{
    static const encode_tiled_fn_t fn = [] {
        void *p = nullptr;
        cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
        const auto err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &status);
#else
        const auto err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &status);
#endif
        if (err != cudaSuccess || status != cudaDriverEntryPointSuccess || p == nullptr) {
            CV_Error(cv::Error::GpuApiCallError, "wgmma_gemm_t: cuTensorMapEncodeTiled not found in the driver");
        }
        return reinterpret_cast<encode_tiled_fn_t>(p);
    }();
    return fn;
}

/**
 * @brief Tensor map of a dimension x dimension row-major half matrix at `ptr` (pitch ld halfs),
 *        read in boxes of box_cols x box_rows with the 128 B swizzle
 */
/// Writes the tensor maps into their final place with generic-proxy stores of this kernel and
/// releases them to the tensor-map proxy: the writer half of the release / acquire pair of PTX
/// (the readers acquire with fence_tensormap()); see make_plan() for why not a cudaMemcpy
__global__ void store_maps_kernel(const CUtensorMap *__restrict__ from, CUtensorMap *__restrict__ to, int count)
{
#if WGMMA_DEVICE_CODE
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        // 128 B per map as 8 x 16 B
        const auto *src = reinterpret_cast<const uint4 *>(from + i);
        auto *dst = reinterpret_cast<uint4 *>(to + i);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dst[k] = src[k];
        }
        asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
    }
#endif
}

CUtensorMap make_map(const half *ptr, int ld, int dimension, int box_cols, int box_rows)
{
    CUtensorMap map{};
    const cuuint64_t dims[2] = {cuuint64_t(dimension), cuuint64_t(dimension)}; // {columns, rows}
    const cuuint64_t strides[1] = {cuuint64_t(ld) * sizeof(half)};               // row pitch, bytes
    const cuuint32_t box[2] = {cuuint32_t(box_cols), cuuint32_t(box_rows)};
    const cuuint32_t element_strides[2] = {1, 1};
    const auto result = encode_tiled()(&map,
                                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                       2,
                                       const_cast<half *>(ptr),
                                       dims,
                                       strides,
                                       box,
                                       element_strides,
                                       CU_TENSOR_MAP_INTERLEAVE_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_128B,
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        std::ostringstream os;
        os << "wgmma_gemm_t: cuTensorMapEncodeTiled failed (" << int(result) << ") for " << ptr << ", ld " << ld;
        CV_Error(cv::Error::GpuApiCallError, os.str());
    }
    return map;
}

/// Device / build check, done once per device (see wgmma_gemm_t::available())
struct availability_t
{
    bool ok = false;
    std::string why;
    int sms = 0;      ///< streaming multiprocessors of the device: the persistent grid without clusters
    int clusters2 = 0; ///< clusters of 2 CTAs that fit at once (cudaOccupancyMaxActiveClusters): the persistent grid with clusters
};

/// The check of the current device: probed once per device (a process may use several GPUs, each
/// thread its own), then served from this thread's last device without a lock
const availability_t &availability()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        device = 0;
    }
    thread_local int cached_device = -1;
    thread_local const availability_t *cached = nullptr;
    if (device == cached_device) {
        return *cached;
    }
    static std::mutex mutex;
    static std::map<int, availability_t> devices; // std::map: references stay valid on insertion
    std::lock_guard<std::mutex> lock(mutex);
    const auto inserted = devices.try_emplace(device);
    auto &a = inserted.first->second;
    if (inserted.second) [&] {
        int major = 0;
        int minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
            || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess
            || cudaDeviceGetAttribute(&a.sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            cudaGetLastError();
            a.why = "no CUDA device";
            return;
        }
        if (major != 9 || minor != 0) {
            a.why = "needs compute capability 9.0 (H100 / H200), this GPU is " + std::to_string(major) + "."
                    + std::to_string(minor);
            return;
        }
        // which code did the runtime load for this GPU? (sm_90a SASS, or PTX of another arch: empty body)
        int *flag = nullptr;
        int value = 0;
        if (cudaMalloc(&flag, sizeof(int)) != cudaSuccess) {
            cudaGetLastError();
            a.why = "cudaMalloc failed";
            return;
        }
        probe_kernel<<<1, 1>>>(flag);
        const auto launched = cudaGetLastError();
        const auto copied = cudaMemcpy(&value, flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(flag);
        if (launched != cudaSuccess || copied != cudaSuccess) {
            cudaGetLastError();
            a.why = "probe kernel failed: no code for this GPU in the build";
            return;
        }
        if (value != 1) {
            a.why = "the build has no sm_90a code (CMake CUDA_ARCH must contain 90a-real)";
            return;
        }
        // every kernel variant: shared memory reserved, registers as the handover needs
        if (!prepare_variant<true, 1, BN>(a.why) || !prepare_variant<false, 1, BN>(a.why)
            || !prepare_variant<true, 2, BN>(a.why) || !prepare_variant<false, 2, BN>(a.why)
            || !prepare_variant<true, 1, TN_SMALL>(a.why) || !prepare_variant<false, 1, TN_SMALL>(a.why)) {
            cudaGetLastError();
            return;
        }
        // how many clusters of 2 fit at once (the GPCs decide, not only the SM count)
        {
            cudaLaunchConfig_t config = {};
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = 2;
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.gridDim = dim3(2 * a.sms);
            config.blockDim = dim3(THREADS);
            config.dynamicSmemBytes = SMEM_BYTES;
            config.attrs = attribute;
            config.numAttrs = 1;
            if (cudaOccupancyMaxActiveClusters(&a.clusters2, wgmma_batched_kernel<false, 2, BN>, &config) != cudaSuccess
                || a.clusters2 < 1) {
                cudaGetLastError();
                a.clusters2 = a.sms / 2; // fall back to the SM count
            }
        }
        a.ok = true;
    }();
    cached_device = device;
    cached = &a;
    return a;
}

} // namespace

bool wgmma_gemm_t::available(std::string &why)
{
    const auto &a = availability();
    why = a.why;
    return a.ok;
}

std::string wgmma_gemm_t::config_name()
{
    std::ostringstream os;
    os << BM << "x" << BN << "x" << BK << " " << STAGES << " stages, TMA + wgmma m64n256k16, "
       << CONSUMERS << " consumer warpgroups, grouped tile order, persistent or one tile per block, clusters of 1 or 2"
          " (B multicast), 128 x 128 tiles (m64n128k16, 6 stages) for small batches,"
          " staged or direct store of C";
    return os.str();
}

wgmma_gemm_t::plan_t wgmma_gemm_t::make_plan(const half *const *A,
                                             int lda,
                                             const half *const *B,
                                             int ldb,
                                             int dimension,
                                             int count,
                                             const void *key)
{
    std::string why;
    if (!available(why)) { // also runs the one-time setup outside any capture
        CV_Error(cv::Error::GpuNotSupported, "wgmma_gemm_t: " + why);
    }
    CV_Assert(supports(dimension));
    CV_Assert(count >= 1 && count <= 65535);
    CV_Assert(lda % 8 == 0 && ldb % 8 == 0 && lda >= dimension && ldb >= dimension);

    std::vector<CUtensorMap> host(2 * size_t(count));
    for (int i = 0; i < count; ++i) {
        CV_Assert(reinterpret_cast<uintptr_t>(A[i]) % 16 == 0 && reinterpret_cast<uintptr_t>(B[i]) % 16 == 0);
        host[2 * i] = make_map(A[i], lda, dimension, BK, BM);             // A: k columns x tile rows
        host[2 * i + 1] = make_map(B[i], ldb, dimension, B_BOX_COLS, BK); // B: 64 columns x k rows
    }
    void *device = nullptr;
    const auto bytes = host.size() * sizeof(CUtensorMap);
    if (cudaMalloc(&device, bytes) != cudaSuccess) {
        cudaGetLastError();
        CV_Error(cv::Error::GpuApiCallError, "wgmma_gemm_t: cudaMalloc of the tensor maps failed");
    }
    // Upload: the host maps go to a staging buffer (cudaMemcpy), then store_maps_kernel() writes
    // them into place with generic stores and fence.proxy.tensormap::generic.release; the kernels
    // acquire them with fence_tensormap(). A plain cudaMemcpy into place is not enough: its
    // copy-engine writes are not released to the tensor-map proxy, and the TMA unit could still
    // use an old descriptor. H100, batched benchmark (a few batched sizes, then 343 leaves of 256):
    // cudaMemcpy crashed 3 / 3 with illegal instructions, also with map addresses never reused
    // (maps never freed) 3 / 3; the kernel upload passed 3 / 3.
    plan_t plan;
    plan.maps = std::shared_ptr<void>(device, [](void *p) { cudaFree(p); });
    void *staging = nullptr;
    if (cudaMalloc(&staging, bytes) != cudaSuccess) {
        cudaGetLastError();
        CV_Error(cv::Error::GpuApiCallError, "wgmma_gemm_t: cudaMalloc of the map staging buffer failed");
    }
    const auto copied = cudaMemcpy(staging, host.data(), bytes, cudaMemcpyHostToDevice);
    const int maps_count = int(host.size());
    store_maps_kernel<<<(maps_count + 127) / 128, 128>>>(static_cast<const CUtensorMap *>(staging),
                                                         static_cast<CUtensorMap *>(device), maps_count);
    const auto stored = cudaDeviceSynchronize(); // the maps are in place before the first launch
    cudaFree(staging);
    if (copied != cudaSuccess || stored != cudaSuccess) {
        cudaGetLastError();
        CV_Error(cv::Error::GpuApiCallError, "wgmma_gemm_t: upload of the tensor maps failed");
    }
    // the round barrier's counter (gemm() resets it before every launch that uses it)
    void *counter = nullptr;
    if (cudaMalloc(&counter, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        CV_Error(cv::Error::GpuApiCallError, "wgmma_gemm_t: cudaMalloc of the round counter failed");
    }
    plan.counter = std::shared_ptr<void>(counter, [](void *p) { cudaFree(p); });
    plan.count = count;
    plan.dimension = dimension;
    plan.key = key;
    return plan;
}

std::string wgmma_gemm_t::last_launch() const
{
    const auto &l = m_last;
    if (l.blocks == 0) {
        return "no launch yet";
    }
    std::ostringstream os;
    os << "grid " << l.blocks << " (" << (l.persistent ? "persistent" : "one unit per cluster") << ", resident "
       << l.resident << (l.cluster > 1 ? " clusters" : " SMs") << "), cluster " << l.cluster
       << (l.attribute ? " attr" : " plain") << ", group " << l.group << " unit rows"
       << ", units " << l.units << ", k tiles " << l.k_tiles
       << (l.round_sync ? ", round sync" : "") << (l.staged ? (l.reduce_targets > 0 ? ", staged reduce-add x" + std::to_string(l.reduce_targets) : std::string(", staged store")) : std::string(", direct store"))
       << ", tile " << BM << "x" << l.tile_n
       << (WGMMA_SETMAXNREG ? "" : ", no setmaxnreg") << (WGMMA_PREFETCH_TENSORMAP ? "" : ", no tensormap prefetch");
    return os.str();
}

void wgmma_gemm_t::gemm(const plan_t &plan, half *const *C, int ldc, accumulator_t accumulator)
{
    // available() was answered by make_plan(): only the cached result is read here (capture safe)
    std::string why;
    if (!available(why)) {
        CV_Error(cv::Error::GpuNotSupported, "wgmma_gemm_t: " + why);
    }
    CV_Assert(plan.valid() && supports(plan.dimension));
    CV_Assert(ldc % 2 == 0 && ldc >= plan.dimension);
    CV_Assert(m_order >= 0);
    const int tiles_m = plan.dimension / BM;
    // tile width (set_tile_n()): automatic = 128 when even the 128-wide tiles (twice as many, each
    // with half the work) all fit on the SMs at once, with cluster 1; else 256.
    // H100, gemm-compare, 128 vs 256 (f16 accumulators): N = 256 5.9 vs 7.4 us, 512 6.9 vs 9.8,
    // 1024 9.2 vs 14.5 (cuBLAS 6.8, 6.8, 12.4); 2048 (128 wide tiles, 256 narrow ones on 132 SMs:
    // two waves) 27.4 vs 24.9, 4096 22.1 vs 18.0
    CV_Assert(m_cluster == 1 || m_cluster == 2);
    int tile_n = m_tile_n;
    if (tile_n == AUTO_TILE_N) {
        const auto wide_tiles = size_t(plan.count) * size_t(tiles_m) * size_t(plan.dimension / BN);
        tile_n = 2 * wide_tiles <= size_t(availability().sms) && m_cluster == 1 ? TN_SMALL : BN;
    }
    CV_Assert(tile_n == BN || (tile_n == TN_SMALL && m_cluster == 1));
    const int tiles_n = plan.dimension / tile_n;
    // schedule: set_schedule(), or automatic: persistent when the whole batch has at least as many
    // tiles as the GPU has SMs. H100, test_diagnostics batched-compare, persistent (automatic g)
    // vs one tile per block, % vs cuBLAS strided batched: 49 x 1024 -0.5 vs +14.8, 343 x 1024
    // -3.1 vs +9.8, 2401 x 1024 -4.0 vs +7.2, 49 / 343 / 2401 x 2048 -15.6 / -12.1 / -10.5 vs
    // about -1, 49 / 343 / 2401 x 512 -7.2 / -2.4 / -4.5 vs +8.6 / +14.8 / +7.2, x 256 within
    // noise. (The first rule looked at one product only - persistent from leaf 4096 - from
    // measurements before the staged store, where many small leaves were faster one tile per block.)
    // Persistent needs at least as many k slices per tile as stages: the producer's 2-entry unit
    // slot, see the kernel; only N = 256 with 128-column tiles has fewer, 4 < 6.
    const auto batch_tiles = size_t(plan.count) * size_t(tiles_m) * size_t(tiles_n);
    const bool persistent_schedule = (m_schedule == schedule_t::persistent
                                      || (m_schedule == schedule_t::automatic && batch_tiles >= size_t(availability().sms)))
                                     && plan.dimension / BK >= stages_of(tile_n);
    const int cluster = m_cluster; // tiles_m is even (M is a multiple of 256), so pairs always fit
    const bool persistent = persistent_schedule;
    // group size of the tile order: set_order(), or automatic (0), measured on the H100
    // (gemm-compare, N = 2048 .. 65536, both schedules):
    //   persistent: 4 tile rows from 32 tile columns on (N >= 8192: one round of 132 blocks is
    //       then a compact 4 x 33 block of tiles; 16384: -4 % vs cuBLAS, 32768: -21 %, where larger
    //       g lost up to 27 %), below that whole tile columns (g = tile rows; 4096: best g 64-128)
    //   one tile per block: half the tile columns, at least 8 (only the first wave keeps the
    //       grouped shape, see the file comment)
    int group = m_order;
    if (group <= 0) {
        if (persistent) {
            group = tiles_n >= 32 ? 4 : tiles_m;
        } else {
            group = std::max(8, tiles_n / 2);
        }
    }
    // the group counts units (cluster tile rows each), at least 1
    group = std::max(1, group / cluster);
    const int group_arg = group;
    // one linear unit index over the whole batch (int: at most 65535 products, see make_plan())
    const auto total = size_t(plan.count) * size_t(tiles_m) * size_t(tiles_n) / size_t(cluster);
    CV_Assert(total <= size_t(INT_MAX));
    const int total_units = int(total);
    // persistent: as many clusters as fit at once (fewer if there are fewer units); else one
    // cluster per unit
    int resident = cluster == 1 ? availability().sms : availability().clusters2;
    if (persistent && m_persistent_blocks > 0) { // set_persistent_blocks(): leave SMs to other kernels
        resident = std::max(1, std::min(resident, m_persistent_blocks / cluster));
    }
    const int blocks = cluster * (persistent ? std::min(resident, total_units) : total_units);
    const auto *maps = static_cast<const CUtensorMap *>(plan.maps.get());
    const int k_tiles = plan.dimension / BK;
    // the round barrier (set_round_sync(), off by default): persistent, cluster 1, static stride,
    // where every CTA is resident at once (a requirement of the spinning wait)
    // on request, or automatically for long tiles (set_round_sync_threshold(), default 512 k slices):
    // H100, gemm-compare: N = 65536 (1024 slices) 651 - 656 ms with the barrier against 750 - 780
    // without (cuBLAS ~698), 32768 (512) +0.7 %, shorter tiles pay for it (2048: +5 %)
    // and automatically for much work per CTA (set_round_sync_work(), default 2^18 k slices x
    // rounds): 49 x 16384 (256 slices, ~3040 rounds) 493 ms with the barrier against 589 without,
    // where the CTAs drift apart and stop sharing their A / B panels in L2
    const int persistent_units = blocks / cluster; // CTAs (clusters) that share the units
    const long long rounds_per_cta = persistent ? (total_units + persistent_units - 1) / persistent_units : 1;
    const bool sync = m_round_sync || (m_round_sync_threshold > 0 && k_tiles >= m_round_sync_threshold)
                      || (m_round_sync_work > 0 && (long long) k_tiles * rounds_per_cta >= m_round_sync_work);
    int *rounds = sync && persistent && cluster == 1 ? static_cast<int *>(plan.counter.get()) : nullptr;
    if (rounds != nullptr) {
        const auto memset_error = cudaMemsetAsync(rounds, 0, sizeof(int), m_stream);
        if (memset_error != cudaSuccess) {
            CV_Error(cv::Error::GpuApiCallError, std::string("wgmma_gemm_t: counter reset failed: ")
                                                     + cudaGetErrorString(memset_error));
        }
    }
    cudaLaunchConfig_t config = {};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = unsigned(cluster);
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.gridDim = dim3(unsigned(blocks));
    config.blockDim = dim3(THREADS);
    config.dynamicSmemBytes = SMEM_BYTES;
    config.stream = m_stream;
    config.attrs = attribute;
    // the cluster attribute only for real clusters: launched with a 1 x 1 x 1 cluster, the
    // persistent kernel ran 1.3 - 2.7x slower than with a plain launch (one tile per block did
    // not change). WGMMA_CLUSTER_ATTR_FOR_1=1 restores that launch, for comparisons.
    static const bool attribute_for_1 = [] {
        const char *v = std::getenv("WGMMA_CLUSTER_ATTR_FOR_1");
        return v != nullptr && std::string(v) == "1";
    }();
    config.numAttrs = (cluster > 1 || attribute_for_1) ? 1 : 0;
    // staged store (set_staged_store()): needs 16 B aligned C rows; ldc here, the product
    // addresses per tile in the kernel
    int staged = m_staged_store && ldc % 8 == 0 ? 1 : 0;
    if (m_reduce_targets > 0) {
        // gemm_reduce(): bulk reduce-add into m_reduce_targets targets per product (staged path only)
        CV_Assert(staged == 1 && m_reduce_targets <= MAX_REDUCE_TARGETS);
        staged = 1 | 2 | (m_reduce_targets << 8);
    }
    // the launch parameters for last_launch(): plain values only, the text is made on request (an
    // ostringstream here cost host time on every call, which bounds the small products' rate)
    m_last.blocks = blocks;
    m_last.persistent = persistent;
    m_last.resident = resident;
    m_last.cluster = cluster;
    m_last.attribute = config.numAttrs != 0;
    m_last.group = group;
    m_last.units = total_units;
    m_last.k_tiles = k_tiles;
    m_last.round_sync = rounds != nullptr;
    m_last.staged = staged != 0;
    m_last.reduce_targets = m_reduce_targets;
    m_last.tile_n = tile_n;
    const auto single = accumulator == accumulator_t::single;
    const cudaError_t launched = single ? launch_for<true>(cluster, tile_n, config, maps, C, ldc, k_tiles, tiles_m, tiles_n,
                                                           group_arg, total_units, rounds, staged)
                                        : launch_for<false>(cluster, tile_n, config, maps, C, ldc, k_tiles, tiles_m, tiles_n,
                                                            group_arg, total_units, rounds, staged);
    const auto err = launched != cudaSuccess ? launched : cudaGetLastError();
    if (err != cudaSuccess) {
        CV_Error(cv::Error::GpuApiCallError, std::string("wgmma_gemm_t: launch failed: ") + cudaGetErrorString(err));
    }
}
