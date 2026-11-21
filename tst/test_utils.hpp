/**
 * @file test_utils.hpp
 * @brief Test utilities and common testing infrastructure
 * @author GEMM Project
 * @date 2025
 * 
 * This header contains utility functions, data structures, and templates
 * used across all test modules for consistent testing and measurement.
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <QCommandLineParser>
#include <QCoreApplication>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <map>

// cuperf_t must be complete: run_test() below is a template that calls its members, and GCC
// rejects that on an incomplete type (MSVC only checked it at instantiation)
#include <perf.hpp>

// Forward declarations
struct stat_t;

/**
 * @namespace GEMM
 * @brief Main namespace for GEMM project functionality
 */
namespace GEMM {

/**
 * @brief Measurement types for performance analysis
 * @details Defines different types of measurements that can be performed
 */
typedef enum {
    TIME_M = 0,    ///< Time-based performance measurement
    ENERGY_M = 1,  ///< Energy consumption measurement
} measurement_t;

// Template structs for measurement titles and units
template<measurement_t m>
struct TITLE_S
{
    static auto text()
    {
        switch (m) {
        case TIME_M:
            return "Performance";
        case ENERGY_M:
            return "Energy Consumption";
        default:
            return "Unknown";
        }
    }
};

template<measurement_t m>
struct UNITS_S
{
    static auto text()
    {
        switch (m) {
        case TIME_M:
            return "time (ms)";
        case ENERGY_M:
            return "energy (mJ)";
        default:
            return "Unknown";
        }
    }
};

/// Runtime units text of a measurement (the CSV y_label)
inline const char *units_text(measurement_t m)
{
    return TIME_M == m ? UNITS_S<TIME_M>::text() : UNITS_S<ENERGY_M>::text();
}

// Test type enums
enum test_id_t {
    CUDA_CAPS = 1,
    CPP_CAPS = 2,
    CUDA_CUBLAS = 3,
    CUDA_CAPS_CUBLAS = 4,
};

enum test_type_t { PERFORMACE = 0, PRECISION = 1 };
enum stat_value_t { DMEAN = 0, DMAX = 1, DSTD = 2 };

// Configuration constants
extern int max_dimension;
extern int max_range;
extern int max_recursions;
extern int warm_ups;
/// run_test(): after the warm_ups calls, keep calling (batches of `iterations` calls, synchronized)
/// until at least this many ms have passed, 0: off. The H100 reaches its power cap (~500 W, clock
/// 1980 -> ~1650 MHz, throttle reason "SW power cap") about 1 s into a load; with only a few warm-up
/// calls the series measured first (cuBLAS) ran at a higher clock than the later ones (N = 8192:
/// up to ~9 %). With a time-based warm-up every series is measured at the steady, capped clock.
extern int warm_up_ms;
extern int iterations;
/// What run_test() measures: TIME_M (ms per call, CUDA events) or ENERGY_M (mJ per call, NVML
/// energy counter, which updates only every tens of ms: meaningful for N >= 8192 only)
extern measurement_t measure;
extern std::string device_name;

// Behavior configuration
extern int keep;
extern bool int_range;
extern bool regenerate;

// Utility functions
/// Whether (N, R) can run here: leaf N >> R >= 256 and BFSCaps::fits_device_memory(N, R) (prints the
/// reason when not)
bool is_valid(int N, int R);
/// Whether A, B and C of N x N fit into this GPU, BFSCaps::fits_device_memory(N, 0); check before
/// allocating a test's inputs (prints the reason when not)
bool inputs_fit(int N);

// Template function for running tests
template<class T>
inline void run_test(cuperf_t& perf, T lambda, cv::cuda::Stream& stream, measurement_t measure_performance)
{
    for (auto i = 0; i < warm_ups; ++i) {
        lambda(stream);
    }
    stream.waitForCompletion();
    if (warm_up_ms > 0) {
        // time-based warm-up (see warm_up_ms): whole batches, the clock read after each batch has
        // finished on the GPU
        const auto start = std::chrono::steady_clock::now();
        const auto budget = std::chrono::milliseconds(warm_up_ms);
        do {
            for (auto i = 0; i < std::max(iterations, 1); ++i) {
                lambda(stream);
            }
            stream.waitForCompletion();
        } while (std::chrono::steady_clock::now() - start < budget);
    }
    // prepare for launch
    switch (measure_performance) {
    case TIME_M: {
        perf.start_timer(stream);
    } break;
    case ENERGY_M: {
        perf.start_monitor_energy(stream);
    } break;
    }
    for (auto i = 0; i < iterations; ++i) {
        lambda(stream);
    }

    switch (measure_performance) {
    case TIME_M: {
        perf.end_timer(stream);
        stream.waitForCompletion();
    } break;
    case ENERGY_M: {
        stream.waitForCompletion();
        perf.end_monitor_energy(stream);
    } break;
    }
    perf.set_count(iterations);
}

// Helper functions
void help(const char* path);

/**
 * @brief Command line options "--name=value" (or "--name value") on top of QCommandLineParser
 *
 *     QCoreApplication app(argc, argv);
 *     GEMM::options_t options("what the program does");
 *     options.add("iterations", "timed iterations", 20);
 *     options.parse(app);                 // handles --help, exits on unknown options
 *     const auto n = options.get_int("iterations");
 */
class options_t
{
public:
    explicit options_t(const char *description);

    options_t &add(const char *name, const char *description, int default_value);
    options_t &add(const char *name, const char *description, const std::string &default_value);

    /// Parses the application's arguments; prints the help and exits for --help or errors
    void parse(const QCoreApplication &app);

    int get_int(const char *name) const;
    std::string get_string(const char *name) const;

private:
    QCommandLineParser m_parser;
    std::vector<QCommandLineOption> m_options;
    const QCommandLineOption &find(const char *name) const;
};

} // namespace GEMM
