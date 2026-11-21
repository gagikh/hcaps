/**
 * @file perf.hpp
 * @brief Timing and energy measurement used by the tests and benchmarks
 * @author GEMM Project
 * @date 2025
 * 
 * Two accumulating measurement objects:
 *  - perf_t: CPU wall-clock timer based on cv::getTickCount(); keeps count / total / min / max
 *    in ticks and reports milliseconds.
 *  - cuperf_t: GPU timer based on a pair of CUDA events recorded on a stream (milliseconds),
 *    which can alternatively accumulate GPU energy read from NVML (millijoules).
 *
 * Both register every live instance in a static set (not thread safe), which lets
 * perf_t::dump() write all CPU timers to perf.csv. Implementation in perf.cpp.
 *
 * Typical GPU benchmark pattern (see tst/performance.cpp, bfs_caps.cu tuning):
 * @code
 *   cuperf_t perf("bfs");
 *   perf.start_timer(stream);
 *   for (int i = 0; i < iterations; ++i) launch(stream);
 *   perf.end_timer(stream);          // one sample covering all iterations
 *   perf.set_count(iterations);      // turn it into a per-iteration average
 *   perf.report(N, std::cerr);       // "bfs, avg, min, max, TFlops=..."
 * @endcode
 */

#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <nvml.h>

#include <set>
#include <fstream>

#include "core.hpp"
#include <cstdint>
#include <iosfwd>
#include <driver_types.h>
#include <ostream>
#include <string>

/**
 * @struct perf_t
 * @brief CPU performance measurement utility
 * 
 * Call start() / end() around the measured code any number of times; every end() adds one
 * sample. All live instances are registered in hinstances_ so that dump() can write them all.
 * Times are stored in ticks (cv::getTickCount) and converted to ms with cv::getTickFrequency.
 * The averages divide by the sample count, so they are undefined before the first end().
 * Not thread safe.
 */
struct perf_t {
    /**
     * @brief Create a timer and register it in the global instance set
     * @param message Label printed in reports and in perf.csv
     */
    perf_t(const std::string &message);

    /**
     * @brief Unregister the timer from the global instance set
     */
    ~perf_t();

    /**
     * @brief Get the average execution time in milliseconds
     * @return Average time over all samples in milliseconds (ticks * 1000 / frequency / count)
     */
    auto get_avg() const {
        const auto freq = cv::getTickFrequency();
        return 1000 * totalTime_ / freq / count_;
    }
    
    /**
     * @brief Get the minimum execution time in milliseconds
     * @return Minimum time in milliseconds
     */
    auto get_min() const {
        const auto freq = cv::getTickFrequency();
        return 1000 * minTime_ / freq;
    }
    
    /**
     * @brief Get the maximum execution time in milliseconds
     * @return Maximum time in milliseconds
     */
    auto get_max() const {
        const auto freq = cv::getTickFrequency();
        return 1000 * maxTime_ / freq;
    }
    
    /**
     * @brief Start timing measurement (remember the current tick count)
     */
    inline void start() {
        startTime_ = cv::getTickCount();
    }

    /**
     * @brief End timing measurement: add one sample (now - start) to count / total / min / max
     */
    inline void end() {
        ++count_;
        const auto diff = cv::getTickCount() - startTime_;
        minTime_ = std::min(minTime_, diff);
        maxTime_ = std::max(maxTime_, diff);
        totalTime_ += diff;
    }

    /**
     * @brief Write all live CPU timers to ./perf.csv (overwritten), one line per timer:
     *        message,count,avg_ms,min_ms,max_ms
     */
    static void dump() {
        std::ofstream outfile;
        outfile.open("perf.csv", std::ios::out | std::ios::trunc);

        for (auto i : perf_t::hinstances_) {
            i->elapsed(outfile);
        }
        outfile.close();
    }

    /**
     * @brief Print "message, avg_ms, min_ms, max_ms" followed by std::endl
     * @param out Output stream
     * @param perf Performance measurement instance
     * @return Reference to output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const perf_t &perf)
    {
        perf.report(out);
        return out;
    }

private:
    /// "message, avg, min, max" in ms (used by operator<<)
    inline void report(std::ostream& out) const {
        auto freq = cv::getTickFrequency();
        out << message_ << ", " << 1000 * totalTime_ / freq / count_ << ", "
            << 1000 * minTime_ / freq << ", " << 1000 * maxTime_ / freq << std::endl;
    }

    /// One CSV line "message,count,avg,min,max" in ms (used by dump())
    inline void elapsed(std::ofstream& out) {
        auto freq = cv::getTickFrequency();
        out << message_ << "," <<
            count_ << "," <<
            1000 * totalTime_ / freq / count_ << "," <<
            1000 * minTime_ /  freq << "," <<
            1000 * maxTime_ / freq << std::endl;
    }

private:
    const std::string message_;     ///< Label used in reports
    uint64_t startTime_ = 0;        ///< Tick count at the last start()
    uint64_t count_ = 0;            ///< Number of samples (end() calls)
    uint64_t totalTime_ = 0;        ///< Sum of all samples, in ticks
    uint64_t minTime_ = 100000000;  ///< Shortest sample in ticks (initial value: large sentinel)
    uint64_t maxTime_ = 0;          ///< Longest sample in ticks
    static std::set<perf_t*> hinstances_; ///< All live perf_t objects, for dump()
};

/**
 * @struct cuperf_t
 * @brief GPU time (CUDA events) or GPU energy (NVML) accumulator
 * @details start_timer() records the start event on a stream; end_timer() records the stop
 *          event on the same stream, waits for it and adds the elapsed time as one sample.
 *          The events measure GPU time between the two points of the stream, not host time.
 *
 *          Alternatively start_monitor_energy() / end_monitor_energy() add the energy used by
 *          the whole GPU in between as a sample (in mJ). Energy and time share the same
 *          accumulators (totalTime_, minTime_, maxTime_), so an object must be used either for
 *          time or for energy, never both.
 *
 *          Can be used from multiple places, but is not thread safe.
 */
struct cuperf_t
{
    /**
     * @brief Create the start / stop events and register the instance
     * @param message Label printed by report()
     */
    cuperf_t(const std::string &message);

    /**
     * @brief Unregister the instance and destroy the events (errors are only logged)
     */
    ~cuperf_t();

    /**
     * @brief Record the start event on @p stream (asynchronous, no host synchronization)
     * @param stream CUDA stream to record events on; end_timer() must use the same stream
     */
    inline void start_timer(cudaStream_t stream) { CHECK(cudaEventRecord(start_, stream)); }
    
    /**
     * @brief Start timing with OpenCV CUDA stream
     * @param stream OpenCV CUDA stream to record events on
     */
    inline void start_timer(cv::cuda::Stream &stream) { start_timer((cudaStream_t) stream.cudaPtr()); }

    /**
     * @brief Read the GPU's cumulative energy counter (NVML) as the start of an energy sample
     * @param stream Unused: the counter is read immediately on the host and covers the whole
     *               device, so the caller must synchronize before calling to bracket the work
     */
    void start_monitor_energy(cudaStream_t stream);
    
    /**
     * @brief Start energy monitoring with OpenCV CUDA stream
     * @param stream OpenCV CUDA stream to monitor
     */
    inline void start_monitor_energy(cv::cuda::Stream &stream)
    {
        start_monitor_energy((cudaStream_t) stream.cudaPtr());
    }

    /**
     * @brief Declare that the recorded sample(s) covered @p val repetitions
     * @details Intended for the pattern "one start/end around a loop of val launches": it
     *          overwrites the sample count with @p val (so get_avg() = total / val) and divides
     *          min and max by @p val. Call it once, after a single start/end pair; after several
     *          end_timer() calls the result is not a meaningful average.
     * @param val Number of repetitions inside the timed window (must be > 0)
     */
    inline void set_count(int val)
    {
        count_ = val;
        minTime_ /= val;
        maxTime_ /= val;
    }

    /**
     * @brief Record the stop event, wait for it (blocks the host) and add the elapsed time
     *        since start_timer() as one sample
     * @param stream CUDA stream to record events on (the one passed to start_timer())
     * @return Elapsed time of this sample in milliseconds
     */
    float end_timer(cudaStream_t stream);
    
    /**
     * @brief End timing and measure elapsed time with OpenCV CUDA stream
     * @param stream OpenCV CUDA stream to record events on
     * @return Elapsed time in milliseconds
     */
    inline auto end_timer(cv::cuda::Stream &stream)
    {
        return end_timer((cudaStream_t) stream.cudaPtr());
    }

    /**
     * @brief Read the energy counter again and add the difference as one sample
     * @param stream Unused (see start_monitor_energy()); synchronize before calling
     * @return Energy of this sample in millijoules, at least 1
     */
    float end_monitor_energy(cudaStream_t stream);
    
    /**
     * @brief End energy monitoring with OpenCV CUDA stream
     * @param stream OpenCV CUDA stream (unused)
     * @return Energy of this sample in millijoules
     */
    inline auto end_monitor_energy(cv::cuda::Stream &stream)
    {
        return end_monitor_energy((cudaStream_t) stream.cudaPtr());
    }

    /**
     * @brief Output stream operator, same as report(out)
     * @param out Output stream
     * @param perf Performance object
     * @return Reference to output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const cuperf_t &perf)
    {
        perf.report(out);
        return out;
    }

    /** @brief Average sample (ms, or mJ for energy): total / count; undefined while count is 0 */
    auto get_avg() const { return totalTime_ / count_; }
    /** @brief Largest sample (divided by the set_count() value, if called) */
    auto get_max() const { return maxTime_; }
    /** @brief Smallest sample (divided by the set_count() value, if called) */
    auto get_min() const { return minTime_; }
    
    /**
     * @brief Throughput of an M x M x M GEMM taking get_avg() milliseconds, in TFLOP/s
     * @details Counts the classical 2 M^3 flops (one multiply + one add per term), also for
     *          Strassen-type algorithms, so it is an "effective" rate comparable with cuBLAS:
     *          2 M^3 / (avg_ms * 1e-3 s) / 1e12 = 2 M^3 * 1e-9 / avg_ms.
     * @param M Matrix dimension
     * @return Effective TFLOP/s
     */
    auto tflops(int M) const
    {
        const auto avg_ms = (double)get_avg();
        return ((double) M) * M * M * 2 * 1e-9 / avg_ms;
    }

    /**
     * @brief Print "message, avg, min, max" followed by std::endl
     * @param out Output stream
     */
    inline void report(std::ostream& out) const {
        out << message_ << ", " << get_avg() << ", " << get_min() << ", " << get_max() << std::endl;
    }

    /**
     * @brief Print "message, avg, min, max, TFlops=<tflops(N)>" followed by std::endl
     * @param N Matrix dimension for TFLOPs calculation
     * @param out Output stream
     */
    inline void report(int N, std::ostream &out) const
    {
        out << message_ << ", " << get_avg() << ", " << get_min() << ", " << get_max()
            << ", TFlops=" << tflops(N) << std::endl;
    }

private:
    /**
     * @brief Initialize NVML and get the handle of device 0 (lazily, on first energy use)
     */
    void init();

private:
    const std::string message_;           ///< Label used in reports
    cudaEvent_t start_, stop_;            ///< Events bracketing one timed window
    uint64_t count_ = 0;                  ///< Number of samples (or the set_count() value)
    float totalTime_ = 0;                 ///< Sum of samples: ms, or mJ when measuring energy
    float minTime_ = 1000000.0f;          ///< Smallest sample (initial value: large sentinel)
    float maxTime_ = 0.0f;                ///< Largest sample
    static std::set<cuperf_t*> cinstances_; ///< All live cuperf_t objects

    static nvmlDevice_t device_;          ///< NVML handle of device 0, nullptr until init()
    unsigned long long energyBefore_ = 0; ///< Energy counter (mJ) at start_monitor_energy()
    unsigned int powerBefore_ = 0;        ///< Currently unused
};
