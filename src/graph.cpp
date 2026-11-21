/**
 * @file graph.cpp
 * @brief Implementation of CUDA graph capture and execution utilities
 * @author GEMM Project
 * @date 2025
 * 
 * This file contains the implementation of the CaptureGraph class that provides
 * utilities for CUDA graph capture and execution, enabling optimized kernel launch patterns.
 * See graph.hpp for the life cycle and the constraints of capturing.
 */

#include <graph.hpp>
#include <core.hpp>

#pragma warning(push, 0)
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#pragma warning(pop)

#include <iostream>

/**
 * @brief Constructor for CUDA graph capture
 * 
 * Initializes the CUDA graph capture object with default settings.
 * Nothing is created here: the graph and its instance come from end_capture().
 */
CaptureGraph::CaptureGraph()
{
}

/**
 * @brief Destructor for CUDA graph capture
 * 
 * Properly destroys CUDA graph objects and execution instances,
 * cleaning up all associated resources.
 */
CaptureGraph::~CaptureGraph()
{
    if (m_graph) {
        auto err = cudaGraphDestroy(m_graph);
        if (cudaSuccess != err) {
            std::cerr << "Graph destroy failure: " << gemm_cuda_error_message(err, "cudaGraphDestroy") << std::endl;
        } else if (REPORT_STATISTICS) {
            std::cerr << "Graph destroy OK" << std::endl;
        }
        m_graph = nullptr;
    }
    if (m_instance) {
        // the executable instance is a separate object from the graph and must be destroyed too
        auto err = cudaGraphExecDestroy(m_instance);
        if (cudaSuccess != err) {
            std::cerr << "Graph instance destroy failure: " << gemm_cuda_error_message(err, "cudaGraphExecDestroy") << std::endl;
        } else if (REPORT_STATISTICS) {
            std::cerr << "Graph instance destroy OK" << std::endl;
        }
        m_instance = nullptr;
    }
}

// Launch the instantiated graph on the specified stream.
// All launch parameters were fixed at instantiation, so a replay costs one API call instead of one
// call per kernel / event in the captured sequence.
bool CaptureGraph::launch(cv::cuda::Stream& stream)
{
    if (nullptr == m_instance) {
        return false;
    }
    auto err = cudaGraphLaunch(m_instance, (cudaStream_t)stream.cudaPtr());
    if (cudaSuccess != err) {
        std::cerr << "Graph launch failure: " << gemm_cuda_error_message(err, "cudaGraphLaunch") << " device " << cv::cuda::getDevice() << std::endl;
        return false;
    } else if (REPORT_STATISTICS) {
        // nothing is printed on success: launch() is on the timed path
        return true;
    }
    return true;
}

// @GH: Important
// Nothing may allocate memory (cudaMalloc, GpuMat::create, thrust vectors, ...) or synchronize
// while the capture is active: with cudaStreamCaptureModeGlobal such a call from any thread
// invalidates the capture, end_capture() then fails and the whole pipeline is unusable.
// All buffers must therefore be allocated before start_capture() (DFSCaps::initialize() does that).
bool CaptureGraph::start_capture(cv::cuda::Stream& stream)
{
    auto err = cudaStreamBeginCapture((cudaStream_t) stream.cudaPtr(), cudaStreamCaptureModeGlobal);
    if (cudaSuccess != err) {
        std::cerr << "Graph capture failure at start: " << gemm_cuda_error_message(err, "cudaStreamBeginCapture") << std::endl;
        return false;
    } else if (REPORT_STATISTICS) {
        std::cerr << "Graph capture start OK" << std::endl;
        return true;
    }
    return true;
}

// Stop capturing, take the recorded graph and instantiate it into an executable graph
// (validation and setup happen once here instead of on every launch).
bool CaptureGraph::end_capture(cv::cuda::Stream& stream, unsigned long long instantiate_flags)
{
    CV_Assert(nullptr == m_instance);
    CV_Assert(nullptr == m_graph);
    auto err = cudaStreamEndCapture((cudaStream_t) stream.cudaPtr(), &m_graph);
    if (cudaSuccess != err) {
        std::cerr << "Graph capture failure at end: " << gemm_cuda_error_message(err, "cudaStreamEndCapture") << std::endl;
        return false;
    } else if (REPORT_STATISTICS) {
        std::cerr << "Graph end capture OK";
    }
    // instantiate_flags: e.g. cudaGraphInstantiateFlagUseNodePriority, so that the kernel nodes keep
    // the priorities of the streams they were captured from (ignored without it)
    err = cudaGraphInstantiateWithFlags(&m_instance, m_graph, instantiate_flags);
    if (cudaSuccess != err) {
        std::cerr << "Graph instantiation failure: " << gemm_cuda_error_message(err, "cudaGraphInstantiate") << std::endl;
        return false;
    } else if (REPORT_STATISTICS) {
        std::cerr << "Graph instantiation OK" << std::endl;
    }
    if (m_instance != nullptr) {
        m_initialized = true;
        return true;
    }
    std::cerr << "Graph instantiation OK but m_instance == nullptr" << std::endl;
    return false;
}
