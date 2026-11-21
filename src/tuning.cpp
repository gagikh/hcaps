/**
 * @file tuning.cpp
 * @brief Lookup in and rewriting of gen/tuning_table.hpp (see include/tuning.hpp)
 *
 * The table is an X-macro file: it is included below with GEMM_TUNED_BFS defined (the table wraps
 * its entries in #ifdef GEMM_TUNED_BFS), and each expansion becomes one row of a static array. Lookups are linear scans; the table has at most a
 * few hundred rows and is read once per initialize().
 */

#include <tuning.hpp>

#include <opencv2/core.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <vector>

namespace tuning {

namespace {

// one GEMM_TUNED_BFS entry
struct bfs_row_t
{
    const char *gpu;
    int N, R;
    bfs_loops_t loops;
};

// The trailing {nullptr, ...} row keeps the array valid when the table has no entries
// (an empty array is ill-formed); the lookups skip it by testing row.gpu.
const bfs_row_t BFS_ROWS[] = {
#define GEMM_TUNED_BFS(gpu, n, r, e, b, rc, d) {gpu, n, r, {e, b, rc, d}},
#include <tuning_table.hpp>
#undef GEMM_TUNED_BFS
    {nullptr, 0, 0, {}}};

} // namespace

const std::string &device_name()
{
    // function-local static: queried once, thread-safe initialization (C++11)
    static const std::string name = [] {
        cudaDeviceProp prop{};
        return cudaSuccess == cudaGetDeviceProperties(&prop, 0) ? std::string(prop.name) : std::string();
    }();
    return name;
}

// Two passes over the rows: first the running GPU's name, then "default" (the order makes a
// GPU's own entry win regardless of where it stands in the table).
bool bfs_loops(const std::string &gpu, int N, int R, bfs_loops_t &loops)
{
    for (const auto *key : {gpu.c_str(), DEFAULT_GPU}) {
        for (const auto &row : BFS_ROWS) {
            if (row.gpu && row.N == N && row.R == R && std::string(row.gpu) == key) {
                loops = row.loops;
                return true;
            }
        }
    }
    return false;
}

void write(const std::string &path,
           const std::string &gpu,
           const bfs_entries_t &bfs,
           const bfs_notes_t &bfs_notes)
{
    // current rows of the file (not of the compiled-in table: the file may be newer)
    std::vector<std::string> bfs_lines;
    {
        // a missing file leaves `in` failed and the loop below reads nothing: the new table then
        // holds only the given entries
        std::ifstream in(path);
        // captures: 1 gpu, 2 N, 3 R (the key; the loop counts are not needed), 4 trailing comment.
        // Only lines in exactly the format written below match; anything else is not kept.
        const std::regex bfs_re(R"re(^GEMM_TUNED_BFS\("(.*)", (\d+), (\d+), \d+, \d+, \d+, \d+\)\s*(//.*)?$)re");
        for (std::string line; std::getline(in, line);) {
            // tolerate a CRLF file (getline keeps the '\r'); the file is written back with '\n'
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            std::smatch m;
            if (std::regex_match(line, m, bfs_re)) {
                // drop this GPU's old entry for a key that is being rewritten; keep everything else
                const auto replaced = m[1] == gpu && bfs.count({std::stoi(m[2]), std::stoi(m[3])});
                if (!replaced) {
                    bfs_lines.push_back(line);
                }
            }
        }
    }
    // the new entries, in the same format the regexes above parse
    for (const auto &e : bfs) {
        std::ostringstream ss;
        ss << "GEMM_TUNED_BFS(\"" << gpu << "\", " << e.first.first << ", " << e.first.second << ", "
           << e.second.encode << ", " << e.second.bilinear << ", " << e.second.reconstruct << ", " << e.second.decode
           << ")";
        const auto note = bfs_notes.find(e.first);
        if (note != bfs_notes.end()) {
            ss << "  // " << note->second;
        }
        bfs_lines.push_back(ss.str());
    }
    // "default" first, then by GPU name; within a GPU by the numbers
    const auto order = [](const std::string &a, const std::string &b) {
        // MACRO("gpu", n1, n2, ...): 1 gpu, 2 the comma-separated numbers
        const std::regex key_re(R"re(^\w+\("(.*)", (.*)\)\s*$)re");
        // the entry without its comment
        const auto code = [](const std::string &line) { return line.substr(0, line.find("//")); };
        const auto ca = code(a), cb = code(b);
        std::smatch ma, mb;
        std::regex_match(ca, ma, key_re);
        std::regex_match(cb, mb, key_re);
        const auto ga = ma[1].str(), gb = mb[1].str();
        if (ga != gb) {
            return ga == DEFAULT_GPU || (gb != DEFAULT_GPU && ga < gb);
        }
        // same GPU: compare all numbers lexicographically, i.e. by (N, R, ...)
        std::istringstream sa(ma[2].str()), sb(mb[2].str());
        std::vector<long> na, nb;
        for (std::string v; std::getline(sa, v, ',');) na.push_back(std::stol(v));
        for (std::string v; std::getline(sb, v, ',');) nb.push_back(std::stol(v));
        return na < nb;
    };
    std::sort(bfs_lines.begin(), bfs_lines.end(), order);

    // everything is in memory now: truncate and write the whole file (header, then the entries
    // inside the #ifdef guard that lets tuning.cpp and bfs_caps.cu include it selectively)
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        CV_Error(cv::Error::StsError, "cannot write " + path);
    }
    out << "// Tuned launch parameters per GPU model (include/tuning.hpp). Written by the tuner:\n"
           "//     ./bin/test_diagnostics tune   (build with -DLOOPS_COUNT_AS_TEMPLATE_ARGS=OFF, then rebuild ON)\n"
           "// which replaces the running GPU's entries and keeps the others. \"default\" covers GPUs without\n"
           "// their own entries. No include guard: included with different macro definitions\n"
           "// (src/tuning.cpp, src/bfs_caps.cu).\n"
           "//\n"
           "//   GEMM_TUNED_BFS(gpu, N, R, encode_loops, bilinear_loops, reconstruct_loops, decode_loops)\n"
           "\n#ifdef GEMM_TUNED_BFS\n";
    for (const auto &l : bfs_lines) {
        out << l << "\n";
    }
    out << "#endif\n";
    if (!out) {
        CV_Error(cv::Error::StsError, "error writing " + path);
    }
}

} // namespace tuning
