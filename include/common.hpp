/**
 * @file common.hpp
 * @brief Shared host-side helpers used by the library, the tests and the benchmarks
 * @author GEMM Project
 * @date 2025
 * 
 * This header is not part of the CAPS pipeline itself; it collects the support code around it:
 *  - device / build reporting: report_device_capabilities(), arch(), print_build_config();
 *  - half <-> float conversion on host and device (to_half / to_float, see layout below);
 *  - test data: urand(), generate_half_data() (random operands, optionally cached on disk),
 *    store() / load() and store_test_result() / load_test_result();
 *  - result checking: stat_t, diff(), calc_diff(), compare();
 *  - an MKL single-precision GEMM used as CPU reference (cv::sgemm);
 *  - cooperative-launch helpers used by the BFS kernels (get_SM_count(),
 *    cooperative_launch_config(), cooperative_launch(), dump_best_config()).
 *
 * Half-packed matrix layout (used everywhere in the project): OpenCV has no fp16 matrix type
 * that GpuMat supports well, so an N x M matrix of `half` is stored as a CV_32FC1 matrix with
 * N rows and M/2 columns. Every float slot holds two consecutive halves of the same row
 * (element (i, 2k) in the low 16 bits, (i, 2k+1) in the high 16 bits, little endian), so
 * row i of the half matrix is simply reinterpret_cast<half*>(mat.ptr<float>(i)).
 * The row step in bytes is the same for both views; only the column count differs.
 *
 * Implementations: common.cpp (host functions) and test_utils.cu (urand and the GpuMat
 * overloads of to_half / to_float).
 */

#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime_api.h> // cudaStream_t (l2_probe)

#include <hmat.hpp>

#include <ostream>
#include <string>

/**
 * @brief Print the properties of CUDA device 0 to std::cerr (memory hierarchy, limits, SM count,
 *        tensor cores, free memory, theoretical bandwidth)
 * @return Device name (cudaDeviceProp::name), used by the benchmarks to label results
 * @note Always queries device 0; the whole project assumes a single GPU.
 * @see common.cpp for implementation
 */
std::string report_device_capabilities();

/**
 * @brief Print the build configuration the library was compiled with (the CMake options) and what
 *        each one means for the kernels; called first in the main() of every GPU test
 * @param out Stream to print to
 */
void print_build_config(std::ostream &out);

/**
 * @brief Get CUDA architecture name from compute capability (only the major version is used)
 * @param major Major compute capability version
 * @param minor Minor compute capability version (printed, not used for the mapping)
 * @return Architecture name, or an empty string for an unknown major version
 * @see common.cpp for implementation
 */
std::string arch(int major, int minor);

/**
 * @brief Color palette for visualization and plotting
 * @details Fixed pseudo-random RGB triples (x, y, z = R, G, B in 0..255) used to give
 *          curves or graph nodes distinct, reproducible colors
 * @note Contains 81 entries; index it modulo the array size. Being `static` in a header,
 *       every translation unit that includes common.hpp gets its own copy.
 */
const static cv::Point3i color_table[]
    = {{89, 161, 197},  {67, 161, 255},  {19, 222, 24},   {186, 55, 2},    {167, 146, 11},
       {190, 76, 98},   {130, 172, 179}, {115, 209, 128}, {204, 79, 135},  {136, 126, 185},
       {209, 213, 45},  {44, 52, 10},    {101, 158, 121}, {179, 124, 12},  {25, 33, 189},
       {45, 115, 11},   {73, 197, 184},  {62, 225, 221},  {32, 46, 52},    {20, 165, 16},
       {54, 15, 57},    {12, 150, 9},    {10, 46, 99},    {94, 89, 46},    {48, 37, 106},
       {42, 10, 96},    {7, 164, 128},   {98, 213, 120},  {40, 5, 219},    {54, 25, 150},
       {251, 74, 172},  {0, 236, 196},   {21, 104, 190},  {226, 74, 232},  {120, 67, 25},
       {191, 106, 197}, {8, 15, 134},    {21, 2, 1},      {142, 63, 109},  {133, 148, 146},
       {187, 77, 253},  {155, 22, 122},  {218, 130, 77},  {164, 102, 79},  {43, 152, 125},
       {185, 124, 151}, {95, 159, 238},  {128, 89, 85},   {228, 6, 60},    {6, 41, 210},
       {11, 1, 133},    {30, 96, 58},    {230, 136, 109}, {126, 45, 174},  {164, 63, 165},
       {32, 111, 29},   {232, 40, 70},   {55, 31, 198},   {148, 211, 129}, {10, 186, 211},
       {181, 201, 94},  {55, 35, 92},    {129, 140, 233}, {70, 250, 116},  {61, 209, 152},
       {216, 21, 138},  {100, 0, 176},   {3, 42, 70},     {151, 13, 44},   {216, 102, 88},
       {125, 216, 93},  {171, 236, 47},  {253, 127, 103}, {205, 137, 244}, {193, 137, 224},
       {36, 152, 214},  {17, 50, 238},   {154, 165, 67},  {114, 129, 60},  {119, 24, 48},
       {73, 8, 110}};

/**
 * @brief Fill a GPU matrix with uniform random numbers in (-range, +range] (asynchronous)
 * @param range Half-width of the interval; values are (2u - 1) * range with u in (0, 1]
 * @param seed Random seed; element (y, x) uses curand subsequence x + y * C.cols, so the
 *             result is deterministic for a given seed and size
 * @param C Output matrix, already allocated, CV_32FC1 or CV_32SC1 (for CV_32SC1 the float
 *          value is truncated to int)
 * @param stream CUDA stream for operations
 * @see test_utils.cu for implementation
 */
extern void urand(int range, unsigned long long seed, cv::cuda::GpuMat &C, cv::cuda::Stream &stream);

/**
 * @brief Convert a square float GPU matrix to half-packed storage (asynchronous)
 * @param A Input N x N CV_32FC1 matrix (must be square)
 * @param AA Output, (re)allocated as N x N/2 CV_32FC1 holding N x N halves (see file header)
 * @param stream CUDA stream for operations
 * @see test_utils.cu for implementation
 */
extern void to_half(const cv::cuda::GpuMat &A, cv::cuda::GpuMat &AA, cv::cuda::Stream &stream);

/**
 * @brief Convert a float host matrix to half-packed storage (round to nearest)
 * @param A Input N x M CV_32FC1 matrix, M must be even (need not be square)
 * @return N x M/2 CV_32FC1 matrix in pinned memory holding the N x M halves (see file header)
 */
cv::HMat to_half(const cv::Mat &A);

/**
 * @brief Convert a half-packed host matrix back to float
 * @param A Input N x M CV_32FC1 matrix holding N x 2M halves
 * @return N x 2M CV_32FC1 matrix (ordinary pageable memory)
 */
cv::Mat to_float(const cv::Mat &A);

/**
 * @brief Convert a half-packed GPU matrix to float, returning a new matrix (asynchronous)
 * @param A Input N x N/2 CV_32FC1 matrix holding N x N halves
 * @param stream CUDA stream for operations
 * @return New N x N CV_32FC1 matrix; valid once @p stream has reached this point
 */
cv::cuda::GpuMat to_float(const cv::cuda::GpuMat& A, cv::cuda::Stream& stream);

/**
 * @brief Convert a half-packed GPU matrix to float into a caller-supplied output (asynchronous)
 * @param A Input N x N/2 CV_32FC1 matrix holding N x N halves (cols * 2 must equal rows)
 * @param AA Output, (re)allocated as N x N CV_32FC1; must not alias @p A
 * @param stream CUDA stream for operations
 */
void to_float(const cv::cuda::GpuMat &A, cv::cuda::GpuMat &AA, cv::cuda::Stream &stream);

/**
 * @brief L2 reuse probe (test_diagnostics l2-reuse): one pass over `bytes` of `buffer` with the
 *        cache operators of the BFS passes (L2_CACHE_POLICY: __stcg / __ldcg), 16 B per thread,
 *        grid-stride, about 8 blocks of 256 threads per SM
 * @param write true: store a pattern (__stcg), false: load everything (__ldcg) and fold it into
 *        `sink` (one int per block, so that the loads are not removed)
 * @param buffer 16 B aligned device memory of at least `bytes` (a multiple of 16)
 * @param sink device int[blocks] for the read pass, blocks = 8 x SMs (may be nullptr for write)
 */
void l2_probe(bool write, void *buffer, size_t bytes, int *sink, int sms, cudaStream_t stream);

/**
 * @brief Bulk store / bulk reduce-add probe (test_diagnostics bulk-reduce): every block keeps one
 *        16 KB tile of halves (all 2^-10) in shared memory and writes it to the 16 KB chunks
 *        blockIdx.x, blockIdx.x + gridDim.x, ... of `dst` with cp.async.bulk (reduce == false) or
 *        cp.reduce.async.bulk .add.noftz.f16 (reduce == true), up to 8 copies in flight per block:
 *        the epilogue operation of a leaf GEMM that adds its tile into its parent's quadrants.
 *        sm_90+ only (the call does nothing on older GPUs).
 * @param dst 16 B aligned device memory, bytes a multiple of 16 KB
 */
void bulk_probe(bool reduce, void *dst, size_t bytes, int sms, cudaStream_t stream);

/**
 * @brief Write a matrix to "../data/<path>" in a raw binary format
 * @details Format: int rows, int cols, int type (OpenCV type code), then rows * cols *
 *          elemSize bytes of data. Despite the ".yaml" names used by the callers, this is not
 *          YAML. The directory is relative to the working directory (the build directory).
 * @param mat Matrix to store; must be continuous (the data block is written in one go)
 * @param path File name relative to ../data/
 * @throws std::runtime_error if the file cannot be opened
 */
void store(const cv::Mat &mat, const cv::String &path);

/**
 * @brief Read a matrix written by store() from "../data/<path>"
 * @param path File name relative to ../data/
 * @return The matrix in pinned memory, or an empty matrix if the file does not exist
 *         (callers use empty() to fall back to regenerating the data)
 */
cv::HMat load(const cv::String &path);

/**
 * @brief Produce the two half-packed N x N test operands (A, B) on the host
 * @details The data are cached in ../data/ under the name
 *          "dimension_<N>_range_<range>_<int|float>_{A,B}.yaml" so that runs are reproducible
 *          and stored golden results stay valid.
 * @param N Matrix dimension
 * @param range Values are uniform in [-range, +range) (cv::randu)
 * @param int_range Draw integers instead of floats; small integers are exactly representable
 *                  in fp16, so rounding of the inputs does not blur the comparison
 * @param regenerate Draw new data instead of loading the cached files
 * @param keep Store newly drawn data to the cache files
 * @return Pair (A, B), each N x N/2 CV_32FC1 in pinned memory holding N x N halves
 * @note If loading fails (no cached file) the data are generated as if @p regenerate was set.
 */
std::pair<cv::HMat, cv::HMat> generate_half_data(
    int N, int range, bool int_range, bool regenerate, bool keep);

/**
 * @brief Same as generate_half_data(N, ...) but the cache files are named
 *        "<test_name>_A.yaml" / "<test_name>_B.yaml"
 * @param test_name Base name of the cache files in ../data/
 * @param N Matrix dimension
 * @param range Values are uniform in [-range, +range)
 * @param int_range Draw integers instead of floats
 * @param regenerate Draw new data instead of loading the cached files
 * @param keep Store newly drawn data to the cache files
 * @return Pair (A, B), each N x N/2 CV_32FC1 in pinned memory holding N x N halves
 */
std::pair<cv::HMat, cv::HMat> generate_half_data(
    const std::string &test_name, int N, int range, bool int_range, bool regenerate, bool keep);

/**
 * @brief Produce the two half-packed N x N test operands directly on the GPU
 * @details New data are drawn on the device with urand() (seeded from std::rand(), so they
 *          differ from the host overload's cv::randu data); cached data use the same file
 *          names as generate_half_data(N, ...), so both overloads share the cache.
 * @param N Matrix dimension
 * @param range Values are uniform in (-range, +range] (urand)
 * @param int_range Draw integers instead of floats (urand into CV_32SC1, truncated toward zero)
 * @param regenerate Draw new data instead of loading the cached files
 * @param keep Download newly drawn data and store them to the cache files
 * @param stream CUDA stream for operations
 * @return Pair (A, B), each N x N/2 CV_32FC1 GpuMat holding N x N halves. On the regenerate
 *         path without @p keep the conversion may still be in flight on @p stream.
 */
std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> generate_half_data(
    int N, int range, bool int_range, bool regenerate, bool keep, cv::cuda::Stream &stream);

/**
 * @brief Store a golden test result as ../data/<test_name>.yaml (raw binary, see store())
 * @param mat Matrix to store
 * @param test_name Test name (will add .yaml extension)
 */
static inline void store_test_result(const cv::Mat& mat, const cv::String& test_name)
{
    store(mat, test_name + ".yaml");
}

/**
 * @brief Load a golden test result stored by store_test_result()
 * @param test_name Test name (will add .yaml extension)
 * @return Loaded matrix (cv::HMat), empty if the file does not exist
 */
static inline auto load_test_result(const cv::String& test_name)
{
    return load(test_name + ".yaml");
}

/**
 * @struct stat_t
 * @brief Statistics of the element-wise absolute difference |A - B| (filled by diff())
 */
struct stat_t
{
    double avgVal = 0.0; ///< Mean of |A - B|
    double maxVal = 0.0; ///< Maximum of |A - B|
    double minVal = 100000.0; ///< Minimum of |A - B| (large initial value, overwritten by diff())
    double stdVal = 0.0; ///< Standard deviation of |A - B| (0 for the GPU overload)
    double relVal = 0.0; ///< ||A - B||_2 / (||B||_2 + 1e-5), B being the reference (0 for the GPU overload)
};

/**
 * @brief Difference statistics between two host matrices
 * @param A Tested matrix
 * @param B Reference matrix (same size and type, single channel)
 * @return All fields of stat_t
 */
stat_t diff(const cv::Mat &A, const cv::Mat &B);

/**
 * @brief Difference statistics between two GPU matrices (synchronizes @p stream)
 * @param A Tested matrix
 * @param B Reference matrix (same size and type, single channel)
 * @param stream CUDA stream for operations
 * @return minVal, maxVal and avgVal only; stdVal and relVal are left at 0
 */
stat_t diff(const cv::cuda::GpuMat &A, const cv::cuda::GpuMat &B, cv::cuda::Stream &stream);

/**
 * @brief Check max |A - B| < eps and print "<msg> passed|failed, max=, mean=, std=" to std::cerr
 * @param A First matrix (usually the reference)
 * @param B Second matrix (same size and type)
 * @param msg Label printed in front of the result
 * @param eps Absolute tolerance on the largest element-wise difference
 * @return true if max |A - B| < eps
 */
bool calc_diff(const cv::Mat &A, const cv::Mat &B, const std::string &msg, double eps);

/**
 * @brief Compare a test output with its golden reference (calc_diff()) and, if it passes and
 *        @p regenerate is set, store it as the new golden result (store_test_result())
 * @param test_name Name of the test (label and golden file name)
 * @param AA First input matrix (unused, kept for the call sites)
 * @param BB Second input matrix (unused, kept for the call sites)
 * @param gold Golden reference result
 * @param out Actual output result
 * @param regenerate Inputs were regenerated, so the stored golden result is stale
 * @param eps Absolute tolerance on max |gold - out|
 * @return true if results match within tolerance, false otherwise
 */
bool compare(const cv::String &test_name,
             const cv::Mat &AA,
             const cv::Mat &BB,
             const cv::Mat &gold,
             const cv::Mat &out,
             bool regenerate,
             double eps);

namespace cv {

/**
 * @brief C = A * B in single precision through MKL (cblas_sgemm, row major); CPU reference
 * @param A First input matrix, square N x N CV_32FC1 (ROIs are fine: the row step is passed)
 * @param B Second input matrix, same size and type as @p A
 * @param C Output matrix, (re)allocated as N x N CV_32FC1; must not alias A or B
 */
void sgemm(const cv::Mat &A, const cv::Mat &B, cv::Mat &C);
}

/**
 * @brief cv::cuda::createContinuous() that also works for 2^31 elements and more
 * @details OpenCV's createContinuous() computes rows * cols as int: from 2^31 elements (N = 65536
 * operands: 65536 x 32768 floats; R = 1 leaf buffer at N = 65536: 7 * 32768 x 16384 floats) it
 * overflows and fails with "The matrix is not continuous, thus its number of rows can not be
 * changed". Below 2^31 elements this calls createContinuous() as before; above, GpuMat::create(),
 * whose pitch equals the row size for our power-of-two widths (rows of 512 B and more), so the
 * buffer is contiguous anyway. Throws if the allocator padded the rows after all.
 * @param rows, cols, type as for cv::cuda::createContinuous()
 * @param mat (re)allocated, rows x cols, step == cols * element size
 */
void create_continuous(int rows, int cols, int type, cv::cuda::GpuMat &mat);

/**
 * @brief Get the number of Streaming Multiprocessors (SMs)
 * @return Number of SMs of device 0
 */
int get_SM_count();

/**
 * @brief Maximum number of blocks of @p fcn that can be co-resident on the GPU
 * @details Occupancy (blocks per SM, assuming no dynamic shared memory) times the SM count.
 *          A cooperative launch (grid-wide sync) is only valid with at most this many blocks.
 * @param fcn Kernel function pointer (the __global__ function cast to void*)
 * @param block Block dimensions
 * @param print Print the numbers to std::cerr (only when REPORT_STATISTICS is enabled)
 * @return Total number of simultaneously active blocks
 */
int cooperative_launch_config(void *fcn, dim3 block, bool print);

/**
 * @brief Print the occupancy-optimal block size and the register usage of @p fcn to std::cerr
 *        (diagnostics only; does nothing unless REPORT_STATISTICS is enabled)
 * @param fcn Kernel function pointer
 */
void dump_best_config(void *fcn);

/**
 * @brief Launch @p fcn with cudaLaunchCooperativeKernel (required for grid.sync())
 * @details Prints a message if the grid exceeds cooperative_launch_config(); the launch is
 *          still attempted and CUDA_CHECK then throws on the resulting error.
 * @param fcn Kernel function pointer
 * @param args Array of pointers to the kernel arguments, in parameter order
 * @param grid Grid dimensions
 * @param block Block dimensions
 * @param stream CUDA stream for operations
 * @note No dynamic shared memory is passed (0 bytes).
 */
void cooperative_launch(void *fcn, void *args[], dim3 grid, dim3 block, cv::cuda::Stream &stream);
