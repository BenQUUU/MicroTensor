#include <gtest/gtest.h>
#include "Tensor.h"
#include "Backend.h"
#include <chrono>
#include <iostream>
#include <random>
#include <cmath>

using namespace mt;

void fill_random(Tensor& t) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dis(gen);
    }
}

TEST(BenchmarkTest, GemmComparison) {
    const int SIZE = 1024;
    
    Tensor A({SIZE, SIZE});
    Tensor B({SIZE, SIZE});
    Tensor bias({SIZE});
    
    Tensor C_base({SIZE, SIZE});
    Tensor C_tiled({SIZE, SIZE});

    fill_random(A);
    fill_random(B);
    fill_random(bias);

    auto start_base = std::chrono::high_resolution_clock::now();
    backend::gemm_cpu_base(A, B, bias, C_base);
    auto end_base = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> base_ms = end_base - start_base;

    auto start_tiled = std::chrono::high_resolution_clock::now();
    backend::gemm_cpu_tiled(A, B, bias, C_tiled);
    auto end_tiled = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiled_ms = end_tiled - start_tiled;

    std::cout << "\n=========================================\n";
    std::cout << "[ BENCHMARK ] Matrix size : " << SIZE << "x" << SIZE << "\n";
    std::cout << "[ BENCHMARK ] Naive CPU (Base): " << base_ms.count() << " ms\n";
    std::cout << "[ BENCHMARK ] Tiling     : " << tiled_ms.count() << " ms\n";
    std::cout << "[ BENCHMARK ] Acceleration  : " << base_ms.count() / tiled_ms.count() << "x\n";
    std::cout << "=========================================\n\n";

#ifdef USE_AVX2
    Tensor C_avx({SIZE, SIZE});
    auto start_avx = std::chrono::high_resolution_clock::now();
    backend::gemm_avx2_tiled(A, B, bias, C_avx);
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_ms = end_avx - start_avx;

    std::cout << "[ BENCHMARK ] AVX2 + Tiling    : " << avx_ms.count() << " ms\n";
    std::cout << "[ BENCHMARK ] AVX2 vs Base     : " << base_ms.count() / avx_ms.count() << "x\n";
#endif

#ifdef USE_CUDA
    Tensor C_cuda({SIZE, SIZE});
    auto start_cuda = std::chrono::high_resolution_clock::now();
    backend::gemm_cuda(A, B, bias, C_cuda);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cuda_ms = end_cuda - start_cuda;

    std::cout << "[ BENCHMARK ] CUDA (Total)     : " << cuda_ms.count() << " ms\n";
    std::cout << "[ BENCHMARK ] CUDA vs AVX2     : " << avx_ms.count() / cuda_ms.count() << "x\n";

    float max_diff_cuda = 0.0f;
    for (size_t i = 0; i < C_base.size(); ++i) {
        float diff = std::abs(C_base.data()[i] - C_cuda.data()[i]);
        if (diff > max_diff_cuda) max_diff_cuda = diff;
    }
    EXPECT_LE(max_diff_cuda, 1e-2f) << "Mathematical discrepancy on GPU!";
#endif

    float max_diff = 0.0f;
    for (size_t i = 0; i < C_base.size(); ++i) {
        float diff = std::abs(C_base.data()[i] - C_tiled.data()[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    EXPECT_LE(max_diff, 1e-3f) << "Mathematical discrepancy!";
}