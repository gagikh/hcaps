/**
 * @file reference_core.hpp
 * @brief Dependency-light CPU reference for the alternative-basis Strassen-Winograd GEMM
 * @author GEMM Project
 * @date 2025
 *
 * Header-only, depends on OpenCV core only (no CUDA, no MKL, no Qt), so it can be
 * built and tested on any machine. The GPU-facing wrappers live in reference.hpp.
 *
 * Notation
 * --------
 * All matrices are square N x N, N a power of two, split recursively into 2x2 blocks
 *
 *         | X11 X12 |
 *     X = |         |     (a, b, c, d) = (X11, X12, X21, X22)
 *         | X21 X22 |
 *
 * down to leaf blocks of size T x T (T = N >> R for R recursions).
 *
 * Alternative basis (Karstadt & Schwartz). Applied once per recursion level; the
 * per-level maps act on disjoint index bits, so the order of levels does not matter.
 *
 *   encode   (inputs A, B):  phi(a, b, c, d)     = (a, b,     c,     b - c + d)
 *   decode   (output C):     psi(a, b, c, d)     = (a, b - d, d - c, d)
 *
 * In the transformed basis one recursion step needs 7 multiplications and only
 * 12 additions:
 *
 *   S1 = A21 + A22   S2 = A22 - A12   S3 = A22 - A11
 *   S4 = B22 - B11   S5 = B21 + B22   S6 = B22 - B12
 *
 *   m1 = A11 * B11   m2 = A12 * B21   m3 = A21 * S4   m4 = A22 * B22
 *   m5 = S1  * S5    m6 = S2  * S6    m7 = S3  * B12
 *
 *   C11 = m1 + m2    C12 = m5 - m7    C21 = m3 + m6   C22 = m5 + m6 - m2 - m4
 *
 * so that   decode(bilinear(encode(A), encode(B))) == A * B.
 *
 * The classic (standard-basis) Strassen-Winograd variant is provided as an
 * independent oracle: strassen_winograd().
 */
#pragma once

#include <opencv2/core.hpp>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace reference {

/// Leaf multiplication C = A * B for T x T blocks. Must accept ROI (non-continuous) inputs.
using leaf_gemm_fn = std::function<void(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)>;

/// Default leaf: OpenCV's own gemm (CV_32F or CV_64F). Slow but has no extra dependencies.
inline void opencv_gemm(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
{
    cv::gemm(A, B, 1.0, cv::noArray(), 0.0, C);
}

// ------------------------------------------------------------------------------------------
// Validation helpers
// ------------------------------------------------------------------------------------------

/// True for 1, 2, 4, ...: a power of two has a single set bit, so x & (x - 1) clears it to 0.
inline bool is_power_of_two(int x) { return x > 0 && 0 == (x & (x - 1)); }

/// Throws cv::Exception unless A is a square, single-channel float/double matrix whose size N
/// is a power of two and the leaf size T is a power of two with T <= N.
inline void check_operand(const cv::Mat &A, int T, const char *who)
{
    const auto fail = [who](const std::string &what) {
        CV_Error(cv::Error::StsBadArg, std::string(who) + ": " + what);
    };
    if (A.empty()) {
        fail("matrix is empty");
    }
    if (A.rows != A.cols) {
        fail("matrix must be square, got " + std::to_string(A.rows) + "x" + std::to_string(A.cols));
    }
    if (A.type() != CV_32FC1 && A.type() != CV_64FC1) {
        fail("matrix must be CV_32FC1 or CV_64FC1");
    }
    if (!is_power_of_two(A.rows)) {
        fail("matrix size must be a power of two, got " + std::to_string(A.rows));
    }
    if (!is_power_of_two(T) || T > A.rows) {
        fail("leaf size T must be a power of two in [1, N], got T=" + std::to_string(T)
             + ", N=" + std::to_string(A.rows));
    }
}

/// Leaf block size for a matrix of size N and R recursions (R = 0 means plain GEMM).
inline int leaf_size(int N, int R)
{
    CV_Assert(R >= 0 && is_power_of_two(N) && (N >> R) >= 1);
    return N >> R;
}

// ------------------------------------------------------------------------------------------
// Block iteration
// ------------------------------------------------------------------------------------------

/// Quadrant views (sharing data with X) of a 2H x 2H block whose top-left corner is column x,
/// row y (cv::Rect convention). Writing into a quadrant writes into X; the views are ROIs and
/// therefore not continuous.
struct quadrants_t
{
    cv::Mat X11, X12, X21, X22;

    quadrants_t(const cv::Mat &X, int x, int y, int H)
        : X11(X(cv::Rect(x, y, H, H)))
        , X12(X(cv::Rect(x + H, y, H, H)))
        , X21(X(cv::Rect(x, y + H, H, H)))
        , X22(X(cv::Rect(x + H, y + H, H, H)))
    {}

    /// The four quadrants of the whole (square, even-sized) matrix X.
    explicit quadrants_t(const cv::Mat &X) : quadrants_t(X, 0, 0, X.cols / 2) {}
};

/// Calls fn(quadrants) for every 2H x 2H block of A, for every level H = T, 2T, ..., N/2.
/// Order: finest level first, blocks row by row. The basis transforms touch disjoint index
/// bits per level, so the result does not depend on this order (tst/reference.cpp checks it).
template<class F>
void for_each_level(int T, cv::Mat &A, F fn)
{
    for (auto H = T; H <= A.cols / 2; H *= 2) {
        for (auto y = 0; y < A.rows; y += 2 * H) {
            for (auto x = 0; x < A.cols; x += 2 * H) {
                quadrants_t q(A, x, y, H);
                fn(q);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Basis transforms (in place)
// ------------------------------------------------------------------------------------------

/// phi: d <- b - c + d at every level. Used for both inputs A and B.
inline void encode(int T, cv::Mat &A)
{
    check_operand(A, T, "reference::encode");
    for_each_level(T, A, [](quadrants_t &q) {
        cv::subtract(q.X22, q.X21, q.X22); // d - c
        cv::add(q.X22, q.X12, q.X22);      // + b
    });
}

/// phi^-1: d <- d - b + c. encode_inverse(T, encode(T, A)) == A.
inline void encode_inverse(int T, cv::Mat &A)
{
    check_operand(A, T, "reference::encode_inverse");
    for_each_level(T, A, [](quadrants_t &q) {
        cv::subtract(q.X22, q.X12, q.X22);
        cv::add(q.X22, q.X21, q.X22);
    });
}

/// psi: b <- b - d, c <- d - c at every level. Maps the product from the alternative
/// basis back to the standard basis.
inline void decode(int T, cv::Mat &C)
{
    check_operand(C, T, "reference::decode");
    for_each_level(T, C, [](quadrants_t &q) {
        cv::subtract(q.X12, q.X22, q.X12); // b <- b - d
        cv::subtract(q.X22, q.X21, q.X21); // c <- d - c
    });
}

/// psi^-1: b <- b + d, c <- d - c. decode_inverse(T, decode(T, C)) == C.
inline void decode_inverse(int T, cv::Mat &C)
{
    check_operand(C, T, "reference::decode_inverse");
    for_each_level(T, C, [](quadrants_t &q) {
        cv::add(q.X12, q.X22, q.X12);
        cv::subtract(q.X22, q.X21, q.X21);
    });
}

// ------------------------------------------------------------------------------------------
// Alternative-basis bilinear step (serial, in place)
// ------------------------------------------------------------------------------------------

namespace detail {

/// A <- A * B in the alternative basis. B is used as scratch space and is destroyed.
/// The operation order mirrors the GPU event schedule (E1..E15) and needs only two
/// H x H temporaries per level.
/// Preconditions: A and B square, same size and type, size = T * 2^k (checked by the callers).
/// A and B may be ROIs; results are always written into their existing buffers.
/// The step numbers below follow the original schedule; E11 covers both m2 and m4, whose
/// operands are untouched by then and which are independent of each other.
inline void alternative_basis_inplace(int T, cv::Mat &A, cv::Mat &B, const leaf_gemm_fn &leaf)
{
    if (A.cols == T) {
        cv::Mat C;
        leaf(A, B, C);
        C.copyTo(A); // A may be an ROI: copyTo keeps writing into the parent buffer
        return;
    }
    quadrants_t a(A);
    quadrants_t b(B);

    // Each quadrant of A / B is overwritten as soon as its last use as an input has passed;
    // the comments give the content after the step.
    cv::Mat temp1, temp2;
    cv::subtract(a.X22, a.X11, temp1);                      // E1:  temp1 = S3
    cv::subtract(b.X22, b.X11, temp2);                      // E2:  temp2 = S4
    alternative_basis_inplace(T, a.X11, b.X11, leaf);       // E3:  A11 = m1
    cv::subtract(b.X22, b.X12, b.X11);                      // E4:  B11 = S6
    alternative_basis_inplace(T, temp1, b.X12, leaf);       // E5:  temp1 = m7
    cv::add(a.X21, a.X22, b.X12);                           // E6:  B12 = S1
    alternative_basis_inplace(T, a.X21, temp2, leaf);       // E7:  A21 = m3
    cv::add(b.X21, b.X22, temp2);                           // E8:  temp2 = S5
    alternative_basis_inplace(T, b.X12, temp2, leaf);       // E9:  B12 = m5
    cv::subtract(a.X22, a.X12, temp2);                      // E10: temp2 = S2
    alternative_basis_inplace(T, a.X12, b.X21, leaf);       // E11: A12 = m2
    alternative_basis_inplace(T, a.X22, b.X22, leaf);       // E11: A22 = m4
    alternative_basis_inplace(T, temp2, b.X11, leaf);       // E12: temp2 = m6

    cv::add(a.X11, a.X12, a.X11);                           // E13: C11 = m1 + m2
    cv::add(a.X21, temp2, a.X21);                           // E14: C21 = m3 + m6
    cv::subtract(b.X12, a.X22, a.X22);                      // E15: C22 = (m5 - m4)
    cv::subtract(temp2, a.X12, temp2);                      //            + (m6 - m2)
    cv::add(a.X22, temp2, a.X22);
    cv::subtract(b.X12, temp1, a.X12);                      //      C12 = m5 - m7
}

} // namespace detail

/**
 * @brief A <- A * B in the alternative basis (A and B already encoded; result still encoded).
 * B is not modified (a scratch copy is taken).
 */
inline void mult_opt(int T, cv::Mat &A, const cv::Mat &B, const leaf_gemm_fn &leaf)
{
    check_operand(A, T, "reference::mult_opt(A)");
    check_operand(B, T, "reference::mult_opt(B)");
    CV_Assert(A.size() == B.size() && A.type() == B.type());
    cv::Mat scratch = B.clone();
    detail::alternative_basis_inplace(T, A, scratch, leaf);
}

/// Full pipeline in the standard basis: returns A * B computed as
/// decode(mult_opt(encode(A), encode(B))). Inputs are not modified.
inline cv::Mat alternative_basis_gemm(int T,
                                      const cv::Mat &A,
                                      const cv::Mat &B,
                                      const leaf_gemm_fn &leaf)
{
    cv::Mat C = A.clone();
    cv::Mat scratch = B.clone();
    encode(T, C);
    encode(T, scratch);
    detail::alternative_basis_inplace(T, C, scratch, leaf);
    decode(T, C);
    return C;
}

// ------------------------------------------------------------------------------------------
// Independent oracle: classic Strassen-Winograd in the standard basis
// ------------------------------------------------------------------------------------------

/// C = A * B using the Winograd variant of Strassen (7 multiplications, 15 additions),
/// recursing down to T x T leaves. Inputs are not modified.
inline void strassen_winograd(int T,
                              const cv::Mat &A,
                              const cv::Mat &B,
                              cv::Mat &C,
                              const leaf_gemm_fn &leaf)
{
    check_operand(A, T, "reference::strassen_winograd(A)");
    check_operand(B, T, "reference::strassen_winograd(B)");
    CV_Assert(A.size() == B.size() && A.type() == B.type());

    if (A.cols == T) {
        leaf(A, B, C);
        return;
    }
    const quadrants_t a(A);
    const quadrants_t b(B);

    const cv::Mat S1 = a.X21 + a.X22;
    const cv::Mat S2 = S1 - a.X11;
    const cv::Mat S3 = a.X11 - a.X21;
    const cv::Mat S4 = a.X12 - S2;
    const cv::Mat T1 = b.X12 - b.X11;
    const cv::Mat T2 = b.X22 - T1;
    const cv::Mat T3 = b.X22 - b.X12;
    const cv::Mat T4 = T2 - b.X21;

    cv::Mat M1, M2, M3, M4, M5, M6, M7;
    strassen_winograd(T, a.X11, b.X11, M1, leaf);
    strassen_winograd(T, a.X12, b.X21, M2, leaf);
    strassen_winograd(T, S4, b.X22, M3, leaf);
    strassen_winograd(T, a.X22, T4, M4, leaf);
    strassen_winograd(T, S1, T1, M5, leaf);
    strassen_winograd(T, S2, T2, M6, leaf);
    strassen_winograd(T, S3, T3, M7, leaf);

    // Winograd's shared partial sums (U1 = M1 + M2 is written directly into C11)
    const cv::Mat U2 = M1 + M6;
    const cv::Mat U3 = U2 + M7;
    const cv::Mat U4 = U2 + M5;

    C.create(A.size(), A.type());
    quadrants_t c(C);
    cv::add(M1, M2, c.X11);
    cv::add(U4, M3, c.X12);      // U5
    cv::subtract(U3, M4, c.X21); // U6
    cv::add(U3, M5, c.X22);      // U7
}

// ------------------------------------------------------------------------------------------
// Layered (BFS) reference: keeps every node so GPU intermediates can be compared
// ------------------------------------------------------------------------------------------

/**
 * @struct test_node_t
 * @brief One node of the recursion tree: its operands, the six sums S1..S6 it produced
 *        and its (encoded) product.
 */
struct test_node_t
{
    /// Keeps headers of A and B (no copy): children of a node are views into the parent's
    /// quadrants and S matrices, so a whole tree shares the buffers of the layer above.
    test_node_t(const cv::Mat &A, const cv::Mat &B) : m_A(A), m_B(B), m_S(6) {}
    const cv::Mat m_A;        ///< Left operand (encoded basis)
    const cv::Mat m_B;        ///< Right operand (encoded basis)
    cv::Mat m_C;              ///< Product m_A * m_B (encoded basis)
    std::vector<cv::Mat> m_S; ///< S1..S6 (empty on the leaf layer)
};

/// Layer index l (0 = root, R = leaves) -> the 7^l nodes of that layer in breadth-first order.
using layers_t = std::map<int, std::vector<test_node_t>>;

/**
 * @brief Breadth-first evaluation of the alternative-basis bilinear algorithm.
 *
 * Layer l holds 7^l nodes, node i of layer l has children 7*i .. 7*i+6 in layer l+1 in the
 * order m1..m7. Layer R is the leaf layer, multiplied with @p leaf.
 * Inputs must already be encoded (see encode()); nodes[0][0].m_C is the encoded product
 * and must be passed through decode() to obtain A * B.
 *
 * Memory: stores every operand, sum and product, i.e. O(N^2 (7/4)^R). Use for moderate sizes.
 * @p leaf has no default on purpose: test_node_t lives in this namespace, so an unqualified
 * 4-argument call would otherwise be ambiguous with ::calculate_layers from reference.hpp.
 */
inline void calculate_layers(int R,
                             const cv::Mat &A,
                             const cv::Mat &B,
                             layers_t &nodes,
                             const leaf_gemm_fn &leaf) // no default: see reference.hpp (ADL)
{
    const auto T = leaf_size(A.rows, R);
    check_operand(A, T, "reference::calculate_layers(A)");
    check_operand(B, T, "reference::calculate_layers(B)");
    CV_Assert(A.size() == B.size() && A.type() == B.type());

    nodes.clear();
    nodes[0].emplace_back(A, B); // root: the whole (encoded) operands

    // forward pass: sums, top-down
    for (auto l = 0; l < R; ++l) {
        auto &next = nodes[l + 1];
        // reserve so emplace_back does not reallocate (node.m_S is referenced by the children
        // only through Mat headers, which are refcounted, but this avoids needless moves)
        next.reserve(7 * nodes[l].size());
        for (auto &node : nodes[l]) {
            const quadrants_t a(node.m_A);
            const quadrants_t b(node.m_B);
            auto &S = node.m_S;
            S[0] = a.X21 + a.X22; // S1
            S[1] = a.X22 - a.X12; // S2
            S[2] = a.X22 - a.X11; // S3
            S[3] = b.X22 - b.X11; // S4
            S[4] = b.X21 + b.X22; // S5
            S[5] = b.X22 - b.X12; // S6

            // children 7*i .. 7*i+6 of node i, in the order m1..m7
            next.emplace_back(a.X11, b.X11); // m1
            next.emplace_back(a.X12, b.X21); // m2
            next.emplace_back(a.X21, S[3]);  // m3
            next.emplace_back(a.X22, b.X22); // m4
            next.emplace_back(S[0], S[4]);   // m5
            next.emplace_back(S[1], S[5]);   // m6
            next.emplace_back(S[2], b.X12);  // m7
        }
    }

    // leaves: T x T products (inputs may be ROIs; the leaf allocates node.m_C)
    for (auto &node : nodes[R]) {
        leaf(node.m_A, node.m_B, node.m_C);
    }

    // backward pass: reconstruct, bottom-up
    for (auto l = R - 1; l >= 0; --l) {
        const auto &next = nodes[l + 1];
        auto idx = 0; // index of the node within layer l; its children start at 7 * idx
        for (auto &node : nodes[l]) {
            const auto &m1 = next[7 * idx + 0].m_C;
            const auto &m2 = next[7 * idx + 1].m_C;
            const auto &m3 = next[7 * idx + 2].m_C;
            const auto &m4 = next[7 * idx + 3].m_C;
            const auto &m5 = next[7 * idx + 4].m_C;
            const auto &m6 = next[7 * idx + 5].m_C;
            const auto &m7 = next[7 * idx + 6].m_C;

            auto &C = node.m_C;
            C.create(node.m_A.size(), node.m_A.type());
            quadrants_t c(C);
            cv::add(m1, m2, c.X11);
            cv::subtract(m5, m7, c.X12);
            cv::add(m3, m6, c.X21);
            // a MatExpr assigned to a Mat would rebind it instead of writing into the ROI,
            // so evaluate into a temporary and copy into the quadrant
            cv::Mat tmp = m5 + m6 - m2 - m4;
            tmp.copyTo(c.X22);
            ++idx;
        }
    }
}

// ------------------------------------------------------------------------------------------
// Error metrics
// ------------------------------------------------------------------------------------------

/// max |A - B|
inline double max_abs_error(const cv::Mat &A, const cv::Mat &B)
{
    return cv::norm(A, B, cv::NORM_INF);
}

/// ||A - B||_F / ||B||_F  (B is the reference); falls back to ||A||_F if B is all zeros
inline double relative_error(const cv::Mat &A, const cv::Mat &B)
{
    const auto denom = cv::norm(B, cv::NORM_L2);
    return denom > 0 ? cv::norm(A, B, cv::NORM_L2) / denom : cv::norm(A, cv::NORM_L2);
}

} // namespace reference
