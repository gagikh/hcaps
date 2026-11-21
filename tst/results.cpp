#include "results.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace GEMM {

std::string results_dir = "../plots/data";

namespace {

/// Start of this run (first result written): the same in every file of the run, py/plots.py
/// --store uses it as the run folder name
const std::string &run_id()
{
    static const std::string id = [] {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return ss.str();
    }();
    return id;
}

std::ofstream open_result(const std::string &name, const std::string &kind, const std::string &title,
                          const std::string &y_label, std::string &path)
{
    std::filesystem::create_directories(results_dir);
    path = (std::filesystem::path(results_dir) / (name + ".csv")).string();
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "cannot write " << path << std::endl;
    }
    out.precision(10);
    out << "# kind: " << kind << "\n"
        << "# title: " << title << "\n"
        << "# y_label: " << y_label << "\n"
        << "# device: " << device_name << "\n";
    cudaDeviceProp prop{};
    if (cudaSuccess == cudaGetDeviceProperties(&prop, 0)) {
        out << "# compute_capability: " << prop.major << "." << prop.minor << "\n"
            << "# sms: " << prop.multiProcessorCount << "\n"
            << "# l2_bytes: " << prop.l2CacheSize << "\n";
    }
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    out << "# cuda_runtime: " << runtime << "\n"
        << "# cuda_driver: " << driver << "\n"
        << "# run: " << run_id() << "\n"
        << "# max_dimension: " << max_dimension << "\n"
        << "# max_recursions: " << max_recursions << "\n"
        << "# max_range: " << max_range << "\n"
        << "# warm_ups: " << warm_ups << "\n"
        << "# warm_up_ms: " << warm_up_ms << "\n"
        << "# iterations: " << iterations << "\n";
    return out;
}

} // namespace

std::string write_series(const std::string &name,
                         const std::string &title,
                         const std::string &y_label,
                         const std::map<std::string, std::map<int, double>> &series)
{
    std::string path;
    auto out = open_result(name, "series", title, y_label, path);
    out << "series,N,value\n";
    for (const auto &s : series) {
        for (const auto &point : s.second) {
            out << s.first << "," << point.first << "," << point.second << "\n";
        }
    }
    std::cerr << "results: " << path << std::endl;
    return path;
}

std::string write_stats(const std::string &name,
                        const std::string &title,
                        const std::string &y_label,
                        std::map<test_type_t, stats_map_t> &stats)
{
    std::string path;
    auto out = open_result(name, "stats", title, y_label, path);
    out << "type,range,R,N,min,max,avg,std,rel\n";
    for (const auto &type : stats) {
        const auto type_name = PERFORMACE == type.first ? "performance" : "precision";
        for (const auto &range : type.second) {
            for (const auto &R : range.second) {
                for (const auto &N : R.second) {
                    const auto &s = N.second;
                    out << type_name << "," << range.first << "," << R.first << "," << N.first << "," << s.minVal
                        << "," << s.maxVal << "," << s.avgVal << "," << s.stdVal << "," << s.relVal << "\n";
                }
            }
        }
    }
    std::cerr << "results: " << path << std::endl;
    return path;
}

} // namespace GEMM
