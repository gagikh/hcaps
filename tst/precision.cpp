#include <common.hpp>
#include <reference.hpp>

#include <bfs_caps.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <vector>
#include <list>
#include <limits>
#include <sstream>
#include <opencv2/core/mat.hpp>
#include <cublas_gemm.hpp>

auto int_range = false;

static bool forward_pass_test(int N, int R, bool regenerate, int range, double eps)
{
    std::stringstream ss;
    ss << __FUNCTION__ << "_" << N << "_" << R;
    auto test_id = ss.str();

    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);
    cv::cuda::GpuMat A_dev, B_dev, C_dev;

    cv::cuda::Stream stream;
    cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
    cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);

    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);
    C_dev.create(AA.size(), AA.type());

    std::map<int, std::vector<test_node_t>> golden_nodes;
    {
        cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
        cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);

        calculate_layers(R, AA, BB, golden_nodes);
    }

    std::map<int, std::vector<std::vector<cv::Mat>>> DS;
    BFSCaps bfs;
    bfs.initialize(A_dev, B_dev, R, stream);
    bfs.bilinear(stream);

    bfs.download_intermediate_sums(DS, stream);

    auto flag = true;
    const auto global_lds = N >> 1;
    for (auto i = 0; i < DS.size(); ++i) {
        const auto& gs = golden_nodes[i];
        const auto& ds = DS[i];
        for (auto j = 0; j < gs.size(); ++j) {
            for (auto k = 0; k < 6; ++k) {
                const auto &g = gs[j].m_S[k];
                const auto &d = ds[j][k];

                std::stringstream ss;
                ss << test_id << "(layer " << i << ", intermediate "
                   << j << ")[" << g.rows << "x" << g.rows << "], S" << k + 1;
                const auto test_name = ss.str();
                {
                    auto dd = to_float(d);
                    flag &= compare(test_name, AA, BB, g, dd, regenerate, 1.0);
                }
            }
        }
    }
    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static bool encoder_test(int N, int R, bool regenerate, int range, double eps)
{
    std::stringstream ss;
    ss << __FUNCTION__ << "_dim=" << N << "_recursions=" << R << "_range=" << range;
    auto test_id = ss.str();

    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);
    auto A = to_float(AA);
    auto B = to_float(BB);

    {
        const auto order = N >> R;
        encode(order, A);
        encode(order, B);
    }

    cv::cuda::GpuMat A_dev, B_dev, C_dev;
    cv::cuda::Stream stream;
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);
    C_dev.create(AA.size(), AA.type());

    std::map<int, std::vector<std::vector<cv::Mat>>> DS;
    BFSCaps bfs;
    bfs.initialize(A_dev, B_dev, R, stream);
    bfs.encode(stream);

    cv::Mat out_A, out_B;
    A_dev.download(out_A, stream);
    B_dev.download(out_B, stream);
    stream.waitForCompletion();
    out_A = to_float(out_A);
    out_B = to_float(out_B);

    auto flag = true;
    flag &= calc_diff(A, out_A, test_id + "_A", eps);
    flag &= calc_diff(B, out_B, test_id + "_B", eps);

    const auto status = flag ? "passed" : "failed";
    std::cerr << test_id + " " << status << std::endl;
    return flag;
}

/// Rounds the float values of X (an ROI view is fine) to the nearest half, ties to even, in place.
static void round_to_half(cv::Mat X)
{
    cv::Mat h;
    X.convertTo(h, CV_16F);
    h.convertTo(X, CV_32F);
}

/// decode(T, C) as the GPU computes it: C holds half values (as float), the levels run top-down
/// (H = N/2 first, as the decoder kernel's l = 1 .. R) and every b - d, d - c is rounded to half
/// (__hsub2 is correctly rounded; a float difference of two halves rounds to the same half).
/// So the result must equal BFSCaps::decode() bit for bit.
static void decode_half_rounded(int T, cv::Mat &C)
{
    for (auto H = C.cols / 2; H >= T; H /= 2) {
        for (auto y = 0; y < C.rows; y += 2 * H) {
            for (auto x = 0; x < C.cols; x += 2 * H) {
                cv::Mat b = C(cv::Rect(x + H, y, H, H));
                cv::Mat c = C(cv::Rect(x, y + H, H, H));
                const cv::Mat d = C(cv::Rect(x + H, y + H, H, H));
                cv::subtract(b, d, b);
                cv::subtract(d, c, c);
                round_to_half(b);
                round_to_half(c);
            }
        }
    }
}

static bool decoder_test(int N, int R, bool regenerate, int range)
{
    std::stringstream ss;
    ss << __FUNCTION__ << "_dim=" << N << "_recursions=" << R << "_range=" << range;
    auto test_id = ss.str();

    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);

    cv::cuda::GpuMat A_dev, B_dev, C_dev;
    cv::cuda::Stream stream;
    A_dev.upload(AA, stream);
    B_dev.upload(BB, stream);

    std::map<int, std::vector<std::vector<cv::Mat>>> DS;
    BFSCaps bfs;
    bfs.initialize(A_dev, B_dev, R, stream);
    bfs.forward_pass(stream);
    bfs.batched_cublas(stream);
    bfs.reconstruct(stream);

    cv::HMat out_A, out_B;
    //cv::Mat src;
    bfs.get_result().download(out_A, stream);
    stream.waitForCompletion();
    {
        const auto order = N >> R;
        out_A = to_float(out_A);
        //out_A.copyTo(src);
        decode_half_rounded(order, out_A);
    }
    bfs.decode(stream);
    bfs.get_result().download(out_B, stream);
    stream.waitForCompletion();
    out_B = to_float(out_B);

    auto flag = true;
    // bit-exact: max |diff| must be 0
    flag &= calc_diff(out_A, out_B, test_id, std::numeric_limits<double>::denorm_min());

    const auto status = flag ? "passed" : "failed";
    std::cerr << test_id + " " << status << std::endl;
    return flag;
}

// Checks the batched cuBLAS leaf products (pointer arrays / leading dimensions set up by
// BFSCaps::prepare_mult_buffers) against one plain cuBLAS GEMM per leaf.
static auto bfs_with_leaf_mult_tests(int N, int R, int range, double eps)
{
    auto regenerate = true;
    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);

    cv::cuda::Stream stream;
    cv::cuda::GpuMat A_dev, B_dev, C_dev;
    {
        cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
        cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);
        C_dev.create(N, N / 2, AA.type());
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);
    }
    auto flag = true;
    {
        BFSCaps bfs;
        bfs.initialize(A_dev, B_dev, R, stream);
        {
            bfs.bilinear(stream);
            bfs.batched_cublas(stream);
        }
        cublas_gemm_t reference_gemm(stream);

        const auto products = bfs.leaf_products(N, R);
        const auto M = N >> R;
        cv::cuda::GpuMat result(M, M / 2, CV_32FC1);
        for (size_t e = 0; e < products.size(); ++e) {
            const auto &p = products[e];
            reference_gemm.gemm(p.L, p.R, result);
            cv::Mat out_M, gold_M;
            result.download(gold_M, stream);
            p.C.download(out_M, stream);
            stream.waitForCompletion();
            std::stringstream ss;
            ss << "Level " << R << " node " << e / MULTS_COUNT << ", mult " << e % MULTS_COUNT;
            const auto O = to_float(out_M);
            const auto G = to_float(gold_M);
            flag &= calc_diff(O, G, ss.str().c_str(), eps);
        }
    }

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}


static auto bfs_with_reconstrcuted_leaf_mult_tests(int N, int R, int range, double eps)
{
    auto regenerate = true;
    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);
    auto A = to_float(AA);
    auto B = to_float(BB);

    cv::Mat C_mkl(N, N, CV_32FC1);
    cv::sgemm(A, B, C_mkl);

    cv::cuda::Stream stream;

    std::stringstream ss;
    ss << __FUNCTION__ << "_dim_" << N << "_recursions_" << R << "_range_" << range;
    const auto test_name = ss.str();
    auto flag = true;

    {
        cv::cuda::GpuMat A_dev, B_dev;
        cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
        cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);

        cublas_gemm_t cgemm(stream);
        cv::cuda::GpuMat C_dev(A_dev.size(), A_dev.type());
        cgemm.gemm(A_dev, B_dev, C_dev);
        cv::Mat C;
        C_dev.download(C, stream);
        stream.waitForCompletion();
        flag &= calc_diff(C_mkl, to_float(C), test_name + "_mkl_cublas", eps);
    }

    const auto order = N >> R;
    BFSCaps bfs;
    cv::cuda::GpuMat A_dev, B_dev;

    std::map<int, std::vector<test_node_t>> golden_nodes;
    {
        encode(order, A);
        encode(order, B);
        AA = to_half(A);
        BB = to_half(B);

        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);
        bfs.initialize(A_dev, B_dev, R, stream);
        calculate_layers(R, AA, BB, golden_nodes);
    }

    if (1) {
        bfs.bilinear(stream);
        bfs.batched_cublas(stream);
        bfs.caps<BFSCaps::RECONSTRUCT_LEAF_MULTS>(stream);
    }

    // test intermediate mults
    std::vector<cv::Mat> outs;
    {
        const auto &address = bfs.get_intermediate_address_cache();
        const auto global_lds = N / 2;

        for (auto l = 1; l < R; ++l) {
            const auto &dst_nodes = address[l - 1];
            const auto &src_nodes = address[l - 0];
            auto start = 0;
            const auto M = N >> l;
            for (auto &m : dst_nodes) {
                cv::Mat C;
                auto dst = cv::cuda::GpuMat(2 * M, M, CV_32FC1, m.m_S, global_lds * sizeof(float));
                dst.download(C, stream);
                stream.waitForCompletion();
                C = to_float(C);
                auto C11 = C.rowRange(0, 1*M).colRange(0, 1 * M);
                auto C12 = C.rowRange(0, 1*M).colRange(M, 2 * M);
                auto C21 = C.rowRange(M, 2*M).colRange(0, 1 * M);
                auto C22 = C.rowRange(M, 2*M).colRange(M, 2 * M);
                outs.push_back(C);

                const auto H = M >> 1;
                std::vector<cv::Mat> result(MULTS_COUNT);
                for (auto i = 0; i < MULTS_COUNT; ++i) {
                    auto iter = src_nodes.begin();
                    std::advance(iter, start + i);
                    auto src = cv::cuda::GpuMat(2 * H, H, CV_32FC1, iter->m_S, global_lds * sizeof(float));
                    cv::Mat host_src;
                    src.download(host_src, stream);
                    stream.waitForCompletion();
                    result[i] = to_float(host_src);
                }
                const auto& m1x = result[0];
                const auto& m2x = result[1];
                const auto& m3x = result[2];
                const auto& m4x = result[3];
                const auto& m5x = result[4];
                const auto& m6x = result[5];
                const auto& m7x = result[6];

                cv::Mat c11 = m1x + m2x;
                cv::Mat c12 = m5x - m7x;
                cv::Mat c21 = m3x + m6x;
                cv::Mat c22 = m5x + m6x - m2x - m4x;

                flag &= calc_diff(C11, c11, test_name + "_C11", eps);
                flag &= calc_diff(C12, c12, test_name + "_C12", eps);
                flag &= calc_diff(C21, c21, test_name + "_C21", eps);
                flag &= calc_diff(C22, c22, test_name + "_C22", eps);
                start += 7;
            }
        }
    }

    {
        std::vector<std::vector<cv::Mat>> HS;
        bfs.download_intermediate_mults(HS, stream);
        auto C3 = to_float(HS[0].front());
        if (!outs.empty()) {
            flag &= calc_diff(C3, outs.front(), test_name + "_FINAL_RESULT", 1e-10);
        }
        decode(order, C3);
        flag &= calc_diff(C_mkl, C3, test_name + "_mkl_vs_CUDA_CAPS", eps);

        {
            auto &C = golden_nodes[0].front().m_C;
            decode(order, C);
            flag &= calc_diff(C_mkl, C, test_name + "_mkl_vs_C++_CAPS", eps);
        }
    }

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static auto full_tests(int N, int R, bool regenerate, int range)
{
    std::stringstream ss;
    ss << __FUNCTION__ << "_dimension_" << N << "_recursions_" << R << "_range_" << range;
    const auto test_name = ss.str();

    auto keep = false;
    auto [AA, BB] = generate_half_data(N, range, int_range, regenerate, keep);

    cv::Mat C_mkl;
    auto A = to_float(AA);
    auto B = to_float(BB);
    cv::sgemm(A, B, C_mkl);

    cv::cuda::Stream stream;
    cv::cuda::GpuMat A_dev, B_dev;

    const auto order = N >> R;
    auto flag = true;
    {
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);

        cv::Mat C_cublas;
        cv::cuda::GpuMat C_dev;
        cublas_gemm_t cgemm(stream);
        cgemm.gemm(A_dev, B_dev, C_dev);
        C_dev.download(C_cublas, stream);
        stream.waitForCompletion();
        C_cublas = to_float(C_cublas);
        flag &= calc_diff(C_mkl, C_cublas, test_name + "_mkl_vs_cublas", 2e-1);
    }

    std::vector<std::vector<cv::Mat>> HS, HM;
    {
        encode(order, A);
        encode(order, B);

        AA = to_half(A);
        BB = to_half(B);
        cv::cuda::createContinuous(AA.size(), AA.type(), A_dev);
        cv::cuda::createContinuous(BB.size(), BB.type(), B_dev);
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);
    }
    std::map<int, std::vector<test_node_t>> golden_nodes;
    if (1)
    {
        calculate_layers(R, AA, BB, golden_nodes);
        auto &C_first = golden_nodes[0].front().m_C;
        decode(order, C_first);
        flag &= calc_diff(C_mkl, C_first, test_name + "_mkl_vs_PARALLELL_C++", 2e-1);
    }

    {
        A_dev.upload(AA, stream);
        B_dev.upload(BB, stream);

        BFSCaps bfs;
        bfs.initialize(A_dev, B_dev, R, stream);
        bfs.bilinear(stream);
        bfs.batched_cublas(stream);
        bfs.reconstruct(stream);
        {
            bfs.download_intermediate_mults(HS, stream);
            auto C3 = to_float(HS[0].front());
            decode(order, C3);
            auto test_id = test_name + "_mkl_vs_CUDA_CAPS+CUBLAS";
            compare(test_id, AA, BB, C_mkl, C3, regenerate, 1.0);
        }
    }

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static auto forward_pass_tests(bool regenerate, int range, double eps)
{
    auto flag = true;

    flag &= forward_pass_test(512, 1, regenerate, range, eps);
    flag &= forward_pass_test(1 * 1024, 1, regenerate, range, eps);
    flag &= forward_pass_test(2 * 1024, 1, regenerate, range, eps);
    flag &= forward_pass_test(4 * 1024, 1, regenerate, range, eps);
    flag &= forward_pass_test(4 * 1024, 2, regenerate, range, eps);
    flag &= forward_pass_test(8 * 1024, 2, regenerate, range, eps);
    flag &= forward_pass_test(8 * 1024, 3, regenerate, range, eps);

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static auto encoder_tests(bool regenerate, int range, double eps)
{
    auto flag = true;

    flag &= encoder_test(512, 1, regenerate, range, eps);
    flag &= encoder_test(1 * 1024, 1, regenerate, range, eps);
    flag &= encoder_test(2 * 1024, 1, regenerate, range, eps);
    flag &= encoder_test(4 * 1024, 1, regenerate, range, eps);
    flag &= encoder_test(4 * 1024, 2, regenerate, range, eps);
    flag &= encoder_test(8 * 1024, 2, regenerate, range, eps);
    flag &= encoder_test(8 * 1024, 3, regenerate, range, eps);
    flag &= encoder_test(16 * 1024, 3, regenerate, range, eps);
    //flag &= encoder_test(16 * 1024, 4, regenerate, range, eps);

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static auto decoder_tests(bool regenerate, int range)
{
    auto flag = true;

    flag &= decoder_test(512, 1, regenerate, range);
    flag &= decoder_test(1 * 1024, 1, regenerate, range);
    flag &= decoder_test(2 * 1024, 1, regenerate, range);
    flag &= decoder_test(4 * 1024, 1, regenerate, range);
    flag &= decoder_test(4 * 1024, 2, regenerate, range);
    flag &= decoder_test(8 * 1024, 2, regenerate, range);
    flag &= decoder_test(8 * 1024, 3, regenerate, range);
    flag &= decoder_test(8 * 1024, 4, regenerate, range);
    flag &= decoder_test(16 * 1024, 3, regenerate, range);
    //flag &= encoder_test(16 * 1024, 4, regenerate, range, eps);

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;
    return flag;
}

static auto leaf_mult_tests(double eps)
{
    auto flag = true;
    {
        const auto N = 4 * 128;
        const auto R = 1;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }
    {
        const auto N = 8 * 128;
        const auto R = 1;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }

    {
        const auto N = 4 * 4 * 128;
        const auto R = 2;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }

    {
        const auto N = 2 * 4 * 128;
        const auto R = 2;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }

    {
        const auto N = 16 * 128;
        const auto R = 2;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }
    {
        const auto N = 16 * 128;
        const auto R = 3;
        flag &= bfs_with_leaf_mult_tests(N, R, 5, eps);
    }
    return flag;
}


static auto reconstructed_leaf_mult_tests(double eps)
{
    auto flag = true;
    if (1) {
        constexpr auto N = 8 * 128;
        constexpr auto R = 1;
        constexpr auto range = 2;
        flag &= bfs_with_reconstrcuted_leaf_mult_tests(N, R, range, 1.0);
    }
    if (1) {
        constexpr auto N = 8 * 128;
        constexpr auto R = 1;
        constexpr auto range = 1;
        flag &= bfs_with_reconstrcuted_leaf_mult_tests(N, R, range, 1.0);
    }
    //if (1) {
    //    constexpr auto N = 16 * 128;
    //    constexpr auto R = 2;
    //    constexpr auto range = 1;
    //    flag &= bfs_with_reconstrcuted_leaf_mult_tests(N, R, range, 1.0);
    //}
    //if (0) {
    //    constexpr auto N = 16 * 128;
    //    constexpr auto R = 3;
    //    constexpr auto range = 1;
    //    flag &= bfs_with_reconstrcuted_leaf_mult_tests(N, R, range, 1.0);
    //}
    return flag;
}

// may be we will work with 32x32 blocks with floats
// each operation will take 2 inputs and 1 outs (lets assume)
// so then 32x32*4 = 4096 bytes for each matrix
// then L2 can hold 1024 matrices
int main(int argc, char *argv[])
{
    auto regenerate = false;
    print_build_config(std::cerr);
    report_device_capabilities();
    srand(1);

    auto flag = true;
    while (true) {

        if (1) {
            flag &= encoder_tests(regenerate, 3, 1e-1);
        }
        if (1) {
            flag &= decoder_tests(regenerate, 3);
        }
        if (1) {
            flag &= forward_pass_tests(regenerate, 3, 1e-1);
        }
        if (1) {
            flag &= leaf_mult_tests(1e-1);
        }
        if (1) {
            reconstructed_leaf_mult_tests(1);
        }
        for (auto range = 1; range < 3; ++range)
        {
            constexpr auto N = 4 * 128;
            constexpr auto R = 1;
            flag &= full_tests(N, R, regenerate, range);
        }

        for (auto range = 1; range < 3; ++range) {
            constexpr auto N = 8 * 128;
            constexpr auto R = 2;
            flag &= full_tests(N, R, regenerate, range);
        }

        for (auto range = 1; range < 3; ++range)
        {
            constexpr auto N = 16 * 128;
            constexpr auto R = 3;
            flag &= full_tests(N, R, regenerate, range);
        }
        for (auto range = 1; range < 3; ++range)
        {
            constexpr auto N = 2*16 * 128;
            constexpr auto R = 4;
            flag &= full_tests(N, R, regenerate, range);
        }
        break;
    }

    const auto status = flag ? "passed" : "failed";
    std::cerr << __FUNCTION__ << " " << status << std::endl;

    cv::HMat::Release();
    return !flag;
}
