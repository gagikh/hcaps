/**
 * @file reference.cpp
 * @brief QtTest suite for the CPU reference (include/reference_core.hpp)
 *
 * Needs neither a GPU nor MKL nor stored test data, so it can run anywhere:
 *     ./bin/test_reference            (all tests)
 *     ./bin/test_reference -v2        (verbose)
 *     ./bin/test_reference encodeRoundTrip
 */

#include <reference_core.hpp>

#include <QtTest/QtTest>

#include <cmath>

namespace {

using namespace reference;

/// Random matrix with a fixed seed. Integer values keep float arithmetic exact for small N.
cv::Mat random_matrix(int N, int type, double low, double high, bool integers, uint64 seed)
{
    cv::RNG rng(seed);
    cv::Mat M(N, N, type);
    if (integers) {
        cv::Mat I(N, N, CV_32SC1);
        rng.fill(I, cv::RNG::UNIFORM, cv::Scalar(low), cv::Scalar(high + 1));
        I.convertTo(M, type);
    } else {
        rng.fill(M, cv::RNG::UNIFORM, cv::Scalar(low), cv::Scalar(high));
    }
    return M;
}

cv::Mat naive_product(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat C;
    cv::gemm(A, B, 1.0, cv::noArray(), 0.0, C);
    return C;
}

/// double-precision product used as ground truth for float computations
cv::Mat exact_product(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat A64, B64;
    A.convertTo(A64, CV_64F);
    B.convertTo(B64, CV_64F);
    return naive_product(A64, B64);
}

double rel_error_vs_exact(const cv::Mat &C, const cv::Mat &exact)
{
    cv::Mat C64;
    C.convertTo(C64, CV_64F);
    return relative_error(C64, exact);
}

bool throws_cv_exception(const std::function<void()> &fn)
{
    try {
        fn();
    } catch (const cv::Exception &) {
        return true;
    }
    return false;
}

/// Adds rows (N, R) for every power-of-two N in [minN, maxN] and recursions R in [0, maxR]
/// with leaf size >= minLeaf.
void add_size_rows(int minN, int maxN, int maxR, int minLeaf)
{
    QTest::addColumn<int>("N");
    QTest::addColumn<int>("R");
    for (auto N = minN; N <= maxN; N *= 2) {
        for (auto R = 0; R <= maxR; ++R) {
            if ((N >> R) < minLeaf) {
                break;
            }
            const auto name = QString("N=%1,R=%2").arg(N).arg(R).toLatin1();
            QTest::newRow(name.constData()) << N << R;
        }
    }
}

} // namespace

class ReferenceTests : public QObject
{
    Q_OBJECT

private slots:
    // --- transforms ------------------------------------------------------------------------
    void encodeSingleBlockMatchesFormula();
    void decodeSingleBlockMatchesFormula();
    void encodeRoundTrip_data() { add_size_rows(2, 256, 8, 1); }
    void encodeRoundTrip();
    void decodeRoundTrip_data() { add_size_rows(2, 256, 8, 1); }
    void decodeRoundTrip();
    void encodeIsLinear();
    void noRecursionIsIdentity();
    void levelsCommute();

    // --- multiplication --------------------------------------------------------------------
    void strassenWinogradExact_data() { add_size_rows(1, 128, 7, 1); }
    void strassenWinogradExact();
    void alternativeBasisExact_data() { add_size_rows(1, 128, 7, 1); }
    void alternativeBasisExact();
    void multOptDoesNotModifyB();
    void multOptWorksOnRoi();
    void alternativeBasisFloatAccuracy_data() { add_size_rows(64, 512, 4, 8); }
    void alternativeBasisFloatAccuracy();
    void identityAndZero();

    // --- layered (BFS) reference -----------------------------------------------------------
    void layersStructure_data() { add_size_rows(4, 64, 3, 2); }
    void layersStructure();
    void layersMatchSerial_data() { add_size_rows(2, 128, 4, 1); }
    void layersMatchSerial();
    void layersSumsFormula();

    // --- validation ------------------------------------------------------------------------
    void rejectsInvalidInput();
};

// ------------------------------------------------------------------------------------------

namespace {

/// 2 x 2 CV_32F matrix [a b; c d]. Not the comma initializer (cv::Mat_<float>(2, 2) << a, b, c, d):
/// OpenCV 5 removed it; this builds the same matrix with OpenCV 4 and 5.
cv::Mat mat2x2(float a, float b, float c, float d)
{
    float values[4] = {a, b, c, d};
    return cv::Mat(2, 2, CV_32F, values).clone(); // clone: own the data, not the local array
}

} // namespace

void ReferenceTests::encodeSingleBlockMatchesFormula()
{
    // phi(a, b, c, d) = (a, b, c, b - c + d)
    cv::Mat A = mat2x2(1, 2, 3, 4);
    encode(1, A);
    const cv::Mat expected = mat2x2(1, 2, 3, 2 - 3 + 4);
    QCOMPARE(max_abs_error(A, expected), 0.0);
}

void ReferenceTests::decodeSingleBlockMatchesFormula()
{
    // psi(a, b, c, d) = (a, b - d, d - c, d)
    cv::Mat C = mat2x2(1, 2, 3, 4);
    decode(1, C);
    const cv::Mat expected = mat2x2(1, 2 - 4, 4 - 3, 4);
    QCOMPARE(max_abs_error(C, expected), 0.0);
}

void ReferenceTests::encodeRoundTrip()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto A0 = random_matrix(N, CV_32F, -8, 8, true, 1);
    cv::Mat A = A0.clone();
    encode(T, A);
    encode_inverse(T, A);
    QCOMPARE(max_abs_error(A, A0), 0.0);
}

void ReferenceTests::decodeRoundTrip()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto C0 = random_matrix(N, CV_32F, -8, 8, true, 2);
    cv::Mat C = C0.clone();
    decode(T, C);
    decode_inverse(T, C);
    QCOMPARE(max_abs_error(C, C0), 0.0);
}

void ReferenceTests::encodeIsLinear()
{
    const auto N = 64, T = 4;
    const auto X = random_matrix(N, CV_64F, -1, 1, false, 3);
    const auto Y = random_matrix(N, CV_64F, -1, 1, false, 4);
    cv::Mat lhs = 2.0 * X - 3.0 * Y;
    encode(T, lhs);
    cv::Mat ex = X.clone(), ey = Y.clone();
    encode(T, ex);
    encode(T, ey);
    const cv::Mat rhs = 2.0 * ex - 3.0 * ey;
    QVERIFY(max_abs_error(lhs, rhs) < 1e-12);
}

void ReferenceTests::noRecursionIsIdentity()
{
    const auto N = 32;
    const auto A0 = random_matrix(N, CV_32F, -5, 5, false, 5);
    cv::Mat A = A0.clone();
    encode(N, A); // T == N: no levels
    decode(N, A);
    QCOMPARE(max_abs_error(A, A0), 0.0);
}

void ReferenceTests::levelsCommute()
{
    // Applying the levels coarse-to-fine must equal fine-to-coarse (the implementation order).
    const auto N = 64, T = 2;
    const auto A0 = random_matrix(N, CV_32F, -8, 8, true, 6);
    cv::Mat fine_first = A0.clone();
    encode(T, fine_first);

    cv::Mat coarse_first = A0.clone();
    for (auto H = N / 2; H >= T; H /= 2) {
        for (auto y = 0; y < N; y += 2 * H) {
            for (auto x = 0; x < N; x += 2 * H) {
                quadrants_t q(coarse_first, x, y, H);
                cv::subtract(q.X22, q.X21, q.X22);
                cv::add(q.X22, q.X12, q.X22);
            }
        }
    }
    QCOMPARE(max_abs_error(fine_first, coarse_first), 0.0);
}

void ReferenceTests::strassenWinogradExact()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto A = random_matrix(N, CV_64F, -4, 4, true, 10 + N);
    const auto B = random_matrix(N, CV_64F, -4, 4, true, 20 + N);
    cv::Mat C;
    strassen_winograd(T, A, B, C, opencv_gemm);
    // small integers in double: every intermediate is exactly representable
    QCOMPARE(max_abs_error(C, naive_product(A, B)), 0.0);
}

void ReferenceTests::alternativeBasisExact()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto A = random_matrix(N, CV_64F, -4, 4, true, 30 + N);
    const auto B = random_matrix(N, CV_64F, -4, 4, true, 40 + N);
    const auto C = alternative_basis_gemm(T, A, B, opencv_gemm);
    QCOMPARE(max_abs_error(C, naive_product(A, B)), 0.0);
}

void ReferenceTests::multOptDoesNotModifyB()
{
    const auto N = 64, T = 8;
    cv::Mat A = random_matrix(N, CV_64F, -4, 4, true, 50);
    const cv::Mat B = random_matrix(N, CV_64F, -4, 4, true, 51);
    const cv::Mat B0 = B.clone();
    encode(T, A);
    mult_opt(T, A, B, opencv_gemm);
    QCOMPARE(max_abs_error(B, B0), 0.0);
}

void ReferenceTests::multOptWorksOnRoi()
{
    // operate on the bottom-right quadrant of a larger buffer; the rest must stay untouched
    const auto N = 32, T = 4;
    const auto big0 = random_matrix(2 * N, CV_64F, -4, 4, true, 60);
    cv::Mat big = big0.clone();
    cv::Mat A = big(cv::Rect(N, N, N, N));
    const auto A0 = A.clone();
    const auto B = random_matrix(N, CV_64F, -4, 4, true, 61);

    cv::Mat eB = B.clone();
    encode(T, A);
    encode(T, eB);
    mult_opt(T, A, eB, opencv_gemm);
    decode(T, A);

    QCOMPARE(max_abs_error(A, naive_product(A0, B)), 0.0);
    QCOMPARE(max_abs_error(big(cv::Rect(0, 0, 2 * N, N)), big0(cv::Rect(0, 0, 2 * N, N))), 0.0);
    QCOMPARE(max_abs_error(big(cv::Rect(0, N, N, N)), big0(cv::Rect(0, N, N, N))), 0.0);
}

void ReferenceTests::alternativeBasisFloatAccuracy()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto A = random_matrix(N, CV_32F, -1, 1, false, 70 + N + R);
    const auto B = random_matrix(N, CV_32F, -1, 1, false, 80 + N + R);
    const auto exact = exact_product(A, B);

    const auto e_naive = rel_error_vs_exact(naive_product(A, B), exact);
    const auto e_alt = rel_error_vs_exact(alternative_basis_gemm(T, A, B, opencv_gemm), exact);
    cv::Mat C_sw;
    strassen_winograd(T, A, B, C_sw, opencv_gemm);
    const auto e_sw = rel_error_vs_exact(C_sw, exact);

    qInfo("N=%d R=%d rel.err: naive=%.2e strassen-winograd=%.2e alternative=%.2e",
          N, R, e_naive, e_sw, e_alt);

    // Strassen-like error grows roughly by a constant factor per level (~ 12^R worst case);
    // stay well above float epsilon but far below anything that would hide a real bug.
    const auto bound = 1e-6 * std::sqrt(double(N)) * std::pow(6.0, R);
    QVERIFY2(e_alt < bound, qPrintable(QString("alt err %1 >= bound %2").arg(e_alt).arg(bound)));
    QVERIFY2(e_sw < bound, qPrintable(QString("sw err %1 >= bound %2").arg(e_sw).arg(bound)));
}

void ReferenceTests::identityAndZero()
{
    const auto N = 64, T = 8;
    const auto A = random_matrix(N, CV_64F, -4, 4, true, 90);
    const cv::Mat I = cv::Mat::eye(N, N, CV_64F);
    const cv::Mat Z = cv::Mat::zeros(N, N, CV_64F);
    QCOMPARE(max_abs_error(alternative_basis_gemm(T, A, I, opencv_gemm), A), 0.0);
    QCOMPARE(max_abs_error(alternative_basis_gemm(T, I, A, opencv_gemm), A), 0.0);
    QCOMPARE(cv::norm(alternative_basis_gemm(T, A, Z, opencv_gemm), cv::NORM_INF), 0.0);
}

void ReferenceTests::layersStructure()
{
    QFETCH(int, N);
    QFETCH(int, R);
    cv::Mat A = random_matrix(N, CV_64F, -4, 4, true, 100);
    cv::Mat B = random_matrix(N, CV_64F, -4, 4, true, 101);
    layers_t nodes;
    calculate_layers(R, A, B, nodes, opencv_gemm);

    QCOMPARE(int(nodes.size()), R + 1);
    auto count = 1;
    for (auto l = 0; l <= R; ++l) {
        QCOMPARE(int(nodes[l].size()), count);
        const auto size = N >> l;
        for (const auto &node : nodes[l]) {
            QCOMPARE(node.m_A.rows, size);
            QCOMPARE(node.m_C.rows, size);
            if (l < R) {
                for (const auto &S : node.m_S) {
                    QCOMPARE(S.rows, size / 2);
                }
            } else {
                QVERIFY(node.m_S[0].empty());
            }
        }
        count *= 7;
    }
}

void ReferenceTests::layersMatchSerial()
{
    QFETCH(int, N);
    QFETCH(int, R);
    const auto T = leaf_size(N, R);
    const auto A = random_matrix(N, CV_64F, -4, 4, true, 110 + N);
    const auto B = random_matrix(N, CV_64F, -4, 4, true, 120 + N);

    cv::Mat eA = A.clone(), eB = B.clone();
    encode(T, eA);
    encode(T, eB);

    layers_t nodes;
    calculate_layers(R, eA, eB, nodes, opencv_gemm);
    cv::Mat C_layers = nodes[0].front().m_C.clone();

    cv::Mat C_serial = eA.clone();
    mult_opt(T, C_serial, eB, opencv_gemm);

    QCOMPARE(max_abs_error(C_layers, C_serial), 0.0); // same algorithm, same encoded result

    decode(T, C_layers);
    QCOMPARE(max_abs_error(C_layers, naive_product(A, B)), 0.0);
}

void ReferenceTests::layersSumsFormula()
{
    // spot-check S1..S6 of the root against the documented formulas
    const auto N = 16, R = 1;
    const auto A = random_matrix(N, CV_64F, -4, 4, true, 130);
    const auto B = random_matrix(N, CV_64F, -4, 4, true, 131);
    layers_t nodes;
    calculate_layers(R, A, B, nodes, opencv_gemm);
    const auto &S = nodes[0].front().m_S;
    const quadrants_t a(A), b(B);
    const cv::Mat expected[6] = {a.X21 + a.X22, a.X22 - a.X12, a.X22 - a.X11,
                                 b.X22 - b.X11, b.X21 + b.X22, b.X22 - b.X12};
    for (auto k = 0; k < 6; ++k) {
        QVERIFY2(max_abs_error(S[k], expected[k]) == 0.0, qPrintable(QString("S%1").arg(k + 1)));
    }
    // children are m1..m7 operands in order
    const auto &child = nodes[1];
    QCOMPARE(max_abs_error(child[0].m_A, a.X11), 0.0);
    QCOMPARE(max_abs_error(child[2].m_B, S[3]), 0.0);
    QCOMPARE(max_abs_error(child[6].m_A, S[2]), 0.0);
    QCOMPARE(max_abs_error(child[6].m_B, b.X12), 0.0);
}

void ReferenceTests::rejectsInvalidInput()
{
    cv::Mat non_pow2(24, 24, CV_32F, cv::Scalar(1));
    cv::Mat non_square(16, 32, CV_32F, cv::Scalar(1));
    cv::Mat wrong_type(16, 16, CV_8U, cv::Scalar(1));
    cv::Mat ok(16, 16, CV_32F, cv::Scalar(1));
    cv::Mat ok_double(16, 16, CV_64F, cv::Scalar(1));
    cv::Mat empty;

    QVERIFY(throws_cv_exception([&] { encode(4, non_pow2); }));
    QVERIFY(throws_cv_exception([&] { encode(4, non_square); }));
    QVERIFY(throws_cv_exception([&] { encode(4, wrong_type); }));
    QVERIFY(throws_cv_exception([&] { encode(4, empty); }));
    QVERIFY(throws_cv_exception([&] { encode(3, ok); }));   // T not a power of two
    QVERIFY(throws_cv_exception([&] { encode(32, ok); }));  // T > N
    QVERIFY(throws_cv_exception([&] { decode(0, ok); }));
    QVERIFY(throws_cv_exception([&] { mult_opt(4, ok, ok_double, opencv_gemm); })); // type mismatch
    QVERIFY(throws_cv_exception([&] {
        layers_t nodes;
        calculate_layers(6, ok, ok, nodes, opencv_gemm); // leaf would be 16 >> 5 == 0
    }));
    QVERIFY(!throws_cv_exception([&] { encode(16, ok); }));
}

QTEST_APPLESS_MAIN(ReferenceTests)
#include "reference.moc"
