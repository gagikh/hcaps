/**
 * @file results.hpp
 * @brief Benchmark results as CSV files for the Python plots (py/plots.py)
 *
 * The benchmarks only measure; py/plots.py draws the graphs (it replaced QCustomPlot).
 * Every file starts with '# key: value' lines, then a CSV table. Keys: kind, title, y_label,
 * device (GPU name), compute_capability, sms, l2_bytes, cuda_runtime, cuda_driver, run (start time
 * of the run, the same in every file of one run) and the benchmark parameters (max_dimension,
 * max_recursions, max_range, warm_ups, iterations). Tables:
 *
 *   series file:  series,N,value                        one line per point of a named curve
 *   stats file:   type,range,R,N,min,max,avg,std,rel    type = performance | precision
 */

#pragma once

#include "test_utils.hpp"

#include <common.hpp>

#include <map>
#include <string>

namespace GEMM {

/// range -> R -> N -> statistics, as filled by the performance / precision tests
using stats_map_t = std::map<int, std::map<int, std::map<int, stat_t>>>;

/// Folder of the result files, default ../plots/data (relative to bin/); --results=<dir>
extern std::string results_dir;

/// Writes <results_dir>/<name>.csv (series file); returns the path
std::string write_series(const std::string &name,
                         const std::string &title,
                         const std::string &y_label,
                         const std::map<std::string, std::map<int, double>> &series);

/// Writes <results_dir>/<name>.csv (stats file) with the performance and precision statistics
std::string write_stats(const std::string &name,
                        const std::string &title,
                        const std::string &y_label,
                        std::map<test_type_t, stats_map_t> &stats);

} // namespace GEMM
