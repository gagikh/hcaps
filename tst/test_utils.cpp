/**
 * @file test_utils.cpp
 * @brief Implementation of test utilities and common testing functions
 * @author GEMM Project
 * @date 2025
 * 
 * This file contains the implementation of utility functions used across
 * all test modules for consistent testing, measurement, and analysis.
 */

#include "test_utils.hpp"
#include <common.hpp>
#include <perf.hpp>
#include <bfs_caps.hpp>
#include <iostream>
#include <string>
#include <algorithm>

namespace GEMM {

// Configuration constants
int max_dimension = 4 * 8192;  ///< Maximum matrix dimension for testing
int max_range = 5;             ///< Maximum range for parameter variation
int max_recursions = 4;        ///< Maximum number of recursions R for testing
int warm_ups = 3;              ///< Number of warm-up iterations
int warm_up_ms = 0;            ///< Time-based warm-up in run_test(), ms (0: off; see test_utils.hpp)
int iterations = 50;           ///< Default number of test iterations
measurement_t measure = TIME_M; ///< --measure=time (default) or energy
std::string device_name = "";  ///< GPU device name for reporting

// Behavior configuration
int keep = 1;                  ///< Keep generated test data
bool int_range = false;        ///< Use integer range for testing
bool regenerate = false;       ///< Regenerate test data

bool is_valid(int N, int R)
{
    // N = k * 2^R with leaves k >= 256 and a multiple of 256 (the 256-half tiles of the passes
    // and of the wgmma leaf GEMM)
    const auto leaf = N >> R;
    if ((leaf << R) != N || leaf < 256 || leaf % 256 != 0) {
        return false;
    }
    // the limits used to be fixed per (N, R) for one GPU; now: what fits into this GPU
    std::string why;
    if (!BFSCaps::fits_device_memory(N, R, &why)) {
        std::cerr << "skip: " << why << std::endl;
        return false;
    }
    return true;
}

bool inputs_fit(int N)
{
    std::string why;
    if (!BFSCaps::fits_device_memory(N, 0, &why)) { // R = 0: A, B and the plain GEMM result
        std::cerr << "skip: " << why << std::endl;
        return false;
    }
    return true;
}

void help(const char* path)
{
    const std::string fullPath(path);
    // Handle both Windows and Unix paths
    const auto lastSlash = fullPath.find_last_of("\\/");
    const auto appName = fullPath.substr(lastSlash + 1);

    std::cerr << "./" << appName << " --max-dim=" << max_dimension
              << " --max-recursions=" << max_recursions << " --max_range=" << max_range
              << " --int-range=" << int_range << " --regenerate=" << regenerate
              << " --iterations=" << iterations << " --warm-ups=" << warm_ups << " --warm-up-ms=" << warm_up_ms
              << " --keep=" << keep << " --measure=" << (TIME_M == measure ? "time" : "energy")
              << std::endl;
}

options_t::options_t(const char *description)
{
    m_parser.setApplicationDescription(description);
    m_parser.addHelpOption();
}

options_t &options_t::add(const char *name, const char *description, int default_value)
{
    return add(name, description, std::to_string(default_value));
}

options_t &options_t::add(const char *name, const char *description, const std::string &default_value)
{
    QCommandLineOption option(name, QString("%1 (default %2)").arg(description, default_value.c_str()), "value",
                              QString::fromStdString(default_value));
    m_parser.addOption(option);
    m_options.push_back(option);
    return *this;
}

void options_t::parse(const QCoreApplication &app)
{
    m_parser.process(app);
}

const QCommandLineOption &options_t::find(const char *name) const
{
    for (const auto &option : m_options) {
        if (option.names().contains(name)) {
            return option;
        }
    }
    CV_Error(cv::Error::StsBadArg, std::string("unknown option ") + name);
}

int options_t::get_int(const char *name) const
{
    bool ok = false;
    const auto text = m_parser.value(find(name));
    const auto value = text.toInt(&ok);
    if (!ok) {
        CV_Error(cv::Error::StsBadArg, "--" + std::string(name) + "=" + text.toStdString() + ": not an integer");
    }
    return value;
}

std::string options_t::get_string(const char *name) const
{
    return m_parser.value(find(name)).toStdString();
}

} // namespace GEMM
