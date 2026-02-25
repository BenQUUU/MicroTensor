#include "Backend.h"

#ifdef USE_AVX2
#include <immintrin.h>
#include <algorithm>

namespace mt {
    namespace backend {
        void gemm_avx2_tiled(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
            int M = A.dim(0);
            int K = A.dim(1);
            int N = B.dim(1);

            const float* a_ptr = A.data();
            const float* b_ptr = B.data();
            const float* bias_ptr = bias.data();
            float* c_ptr = C.data();

            #pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                int j = 0;
                if (bias_ptr) {
                    for (; j <= N - 8; j += 8) {
                        __m256 v_bias = _mm256_loadu_ps(bias_ptr + j);
                        _mm256_storeu_ps(c_ptr + i * N + j, v_bias);
                    }
                    for (; j < N; ++j) c_ptr[i * N + j] = bias_ptr[j];
                } else {
                    for (; j <= N - 8; j += 8) {
                        _mm256_storeu_ps(c_ptr + i * N + j, _mm256_setzero_ps());
                    }
                    for (; j < N; ++j) c_ptr[i * N + j] = 0.0f;
                }
            }

            constexpr int BLOCK_SIZE = 64;

            #pragma omp parallel for collapse(2)
            for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
                for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                    for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {

                        int i_max = std::min(i0 + BLOCK_SIZE, M);
                        int k_max = std::min(k0 + BLOCK_SIZE, K);
                        int j_max = std::min(j0 + BLOCK_SIZE, N);

                        for (int i = i0; i < i_max; ++i) {
                            for (int k = k0; k < k_max; ++k) {
                                __m256 v_a = _mm256_set1_ps(a_ptr[i * K + k]);

                                int j = j0;
                                for (; j <= j_max - 8; j += 8) {
                                    __m256 v_b = _mm256_loadu_ps(b_ptr + k * N + j);
                                    __m256 v_c = _mm256_loadu_ps(c_ptr + i * N + j);

                                    v_c = _mm256_fmadd_ps(v_a, v_b, v_c);

                                    _mm256_storeu_ps(c_ptr + i * N + j, v_c);
                                }

                                for (; j < j_max; ++j) {
                                    c_ptr[i * N + j] += a_ptr[i * K + k] * b_ptr[k * N + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif