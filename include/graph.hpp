/**
 * @file graph.hpp
 * @brief CUDA graph capture and execution utilities
 * @author GEMM Project
 * @date 2025
 * 
 * This header contains the CaptureGraph class that provides utilities for
 * CUDA graph capture and execution, enabling optimized kernel launch patterns.
 *
 * Used by DFSCaps: initialize() records its whole run (encode, per root product the operand forming,
 * sub-tree sums and batched leaf GEMM, on one or two streams joined by events, then decode) once
 * into a graph, and run() replays it with one cudaGraphLaunch(), replacing dozens of individual
 * launches and the host-side event logic by a single call.
 *
 * Life cycle: start_capture() -> issue work on the stream (and on streams that join it through
 * events) -> end_capture() (builds and instantiates the graph) -> launch() any number of times.
 * A CaptureGraph can be captured only once (end_capture() asserts that no graph exists yet).
 *
 * The graph stores the kernel arguments and device addresses as they were at capture time, so the
 * buffers it uses must stay allocated and at the same addresses for as long as it is launched.
 */

#pragma once

#pragma warning(push, 0)
#include <driver_types.h>
#pragma warning(pop)

#include <opencv2/core/cuda.hpp>

/**
 * @class CaptureGraph
 * @brief CUDA graph capture and execution wrapper
 * 
 * Provides a convenient interface for CUDA graph capture and execution,
 * enabling optimized kernel launch patterns and reduced CPU overhead.
 * Errors are reported on std::cerr and as a false return value (no exceptions), except the
 * assertions of end_capture().
 */
class CaptureGraph
{
    public:
        /**
         * @brief Constructor for CUDA graph capture
         */
        CaptureGraph();
        
        /**
         * @brief Destructor for CUDA graph capture: destroys the graph and its executable instance
         *        (errors are only logged)
         */
        ~CaptureGraph();

        /**
         * @brief Start CUDA graph capture
         * @param stream CUDA stream to capture; it must not be the legacy default stream
         *        (capture of stream 0 is not allowed)
         * @return true if capture started successfully
         * @details Global capture mode: until end_capture(), "unsafe" CUDA calls such as
         *          cudaMalloc or synchronous copies from any thread invalidate the capture.
         */
        bool start_capture(cv::cuda::Stream& stream);
        
        /**
         * @brief End CUDA graph capture
         * @param stream CUDA stream being captured
         * @return true if capture ended successfully
         * @details Also instantiates the graph (cudaGraphInstantiate). Every stream forked from
         *          the capture stream must have been joined back (the capture stream waited on an
         *          event recorded after its last work), otherwise capture fails.
         * @pre No graph was captured before by this object (CV_Assert).
         */
        bool end_capture(cv::cuda::Stream& stream, unsigned long long instantiate_flags = 0);
        
        /**
         * @brief Launch the captured CUDA graph
         * @param stream CUDA stream to launch on (any stream, not necessarily the captured one)
         * @return true if launch was successful; false also when nothing has been captured yet
         * @details Asynchronous: the graph runs in stream order after the work already queued.
         */
        bool launch(cv::cuda::Stream& stream);
        
        /**
         * @brief Check if graph is initialized and ready
         * @return true if graph is initialized (end_capture() succeeded)
         */

    private:
        cudaGraph_t m_graph = nullptr;        ///< CUDA graph object
        cudaGraphExec_t m_instance = nullptr; ///< CUDA graph execution instance
        bool m_initialized = false;           ///< Initialization status
};
