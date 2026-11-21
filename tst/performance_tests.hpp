/**
 * @file performance_tests.hpp
 * @brief Performance testing functions for GEMM implementations
 * @author GEMM Project
 * @date 2025
 * 
 * This header contains declarations for performance testing functions that
 * benchmark different GEMM implementations including CUBLAS and CAPS+CUBLAS.
 */

#pragma once

#include <map>
#include "test_utils.hpp"

namespace GEMM {

/**
 * @brief Run comprehensive CUBLAS performance tests
 * @param statistcs Output map to store performance statistics
 * 
 * Executes performance tests for CUBLAS GEMM implementation across
 * different matrix dimensions and configurations. Results are stored
 * in the provided statistics map for analysis and visualization.
 */
void run_cublas_performance_tests(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs);

/**
 * @brief Run CAPS+CUBLAS performance tests
 * @param statistcs Output map to store performance statistics
 * 
 * Executes performance tests for CAPS+CUBLAS combined GEMM implementation
 * across different matrix dimensions and configurations. Results are stored
 * in the provided statistics map for comparison with other implementations.
 */
void run_caps_cublas_performance_tests(
    std::map<GEMM::test_type_t, std::map<int, std::map<int, std::map<int, stat_t>>>> &statistcs);

} // namespace GEMM
