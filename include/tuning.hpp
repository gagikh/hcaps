/**
 * @file tuning.hpp
 * @brief Tuned launch parameters per GPU model, compiled in from gen/tuning_table.hpp
 *
 * gen/tuning_table.hpp is written by the tuner (./bin/test_diagnostics tune, in a build with
 * -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF) and holds, per GPU (the CUDA device name):
 *
 *   GEMM_TUNED_BFS(gpu, N, R, encode, bilinear, reconstruct, decode)
 *       loop counts of the BFS kernels: the select_kernel() table of bfs_caps.cu (compiled into the
 *       kernels in the default build) and the starting configuration of BFSCaps::m_config
 *
 * Entries of the running GPU win; "default" entries cover everything else. Re-running the tuner
 * on a GPU replaces that GPU's entries and keeps all others, so one table serves every model.
 *
 * Users: BFSCaps::initialize() (bfs_loops() -> BFSCaps::m_config), select_kernel() in bfs_caps.cu (includes
 * the table itself with its own macro definitions) and the tuner in tst/diagnostics.cpp (write()).
 *
 * The lookup matches exactly: an (N, R) without an entry for this GPU
 * nor for "default" returns false and the caller falls back to its own heuristic. The key is the
 * device name string, so all boards of one model share entries.
 */

#pragma once

#include <map>
#include <string>

namespace tuning {

/// "default": the entries used for GPUs without their own
constexpr const char *DEFAULT_GPU = "default";

/// Loop counts of the four BFS passes for one (N, R): each pass processes its rows in this many
/// iterations of its outer row loop (rows per iteration = rows / count, see BFSCaps::config_t; each
/// iteration ends in a grid-wide barrier). 1 = the whole pass in one iteration.
struct bfs_loops_t
{
    int encode = 1, bilinear = 1, reconstruct = 1, decode = 1;
};

/// Name of CUDA device 0 (cudaDeviceProp::name, e.g. "NVIDIA GeForce RTX 4090"), the key of the
/// table. Queried once and cached for the process; an empty string if the query fails (then only
/// "default" entries match). Always device 0, not the current device: multi-GPU runs on another
/// device use device 0's entries.
const std::string &device_name();

/// Loop counts for (N, R) on gpu, else the "default" entry; false if there is none
bool bfs_loops(const std::string &gpu, int N, int R, bfs_loops_t &loops);

using bfs_entries_t = std::map<std::pair<int, int>, bfs_loops_t>;              ///< (N, R) -> loops
using bfs_notes_t = std::map<std::pair<int, int>, std::string>;                ///< (N, R) -> comment

/// Rewrites the table file at path: the entries of gpu for the given keys are replaced or added,
/// every other line is kept (with its comment). A new entry is followed by "// note" when the
/// notes have one for its key (the tuner writes the measured times there). Throws cv::Exception
/// on I/O errors.
/// "Every other line" means every other GEMM_TUNED_BFS entry in the exact format write() produces;
/// the header comment and the #ifdef / #endif lines are regenerated, and lines in any other format
/// (e.g. hand-edited with different spacing) are dropped. The entries are sorted: "default" first,
/// then by GPU name, then numerically by key. A missing file is treated as empty. The new table
/// takes effect only after rebuilding (it is compiled in).
void write(const std::string &path,
           const std::string &gpu,
           const bfs_entries_t &bfs,
           const bfs_notes_t &bfs_notes);

} // namespace tuning
