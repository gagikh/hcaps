/**
 * @file bfs_caps.cuh
 * @brief Declarations of the constant-memory node address tables of the CAPS kernels
 * @author GEMM Project
 * @date 2025
 * 
 * This header contains CUDA-specific definitions and declarations for the BFS GEMM
 * algorithm: the extern declarations of the two constant-memory tables through which the device
 * code (bilinear.cuh, reconstruct.cuh, bfs_kernel in bfs_caps.cu) finds the nodes of the recursion
 * tree. They are defined in bfs_caps.cu and filled by BFSCaps::store_address_cache() during initialize().
 * Included by bfs_caps.cu and bilinear.cuh.
 */

#pragma once

#include "bfs_caps.hpp"
#include "core.hpp"

// 3 float pointers in each, 3 * 8 = 24 bytes
/** @brief Internal nodes (levels 0 .. R - 1) in breadth-first order, level after level: operands
 *         m_L, m_R and S area m_S of each node; children of node i of a level start at
 *         start(next level) + 7 * i */
extern __constant__ BFSCaps::layer_info_t __intermediate_address_cache__[ADDRESS_CACHE_SIZE];
/** @brief Per node of level R - 1 (same order): start of its 7 leaf products, back to back,
 *         each (N >> R) x (N >> R) halves with leading dimension N >> R */
extern __constant__ float *__mult_address_cache__[MULT_CACHE_SIZE];
