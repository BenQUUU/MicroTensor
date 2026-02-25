#include "Backend.h"
#include <algorithm>

namespace mt {
    namespace backend {
        void gemm_cpu_base(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
            // A: [M, K]  -> M = Batch Size, K = Input Features
            // B: [K, N]  -> K = Input Features, N = Output Features
            // C: [M, N]

            int M = A.dim(0);
            int K = A.dim(1);
            int N = B.dim(1);

            const float* a_ptr = A.data();
            const float* b_ptr = B.data();
            const float* bias_ptr = bias.data();
            float* c_ptr = C.data();

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {

                    float sum = (bias_ptr != nullptr) ? bias_ptr[j] : 0.0f;

                    for (int k = 0; k < K; ++k) {
                        sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                    }

                    c_ptr[i * N + j] = sum;
                }
            }
        }

        void gemm_cpu_tiled(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
            int M = A.dim(0);
            int K = A.dim(1);
            int N = B.dim(1);

            const float* a_ptr = A.data();
            const float* b_ptr = B.data();
            const float* bias_ptr = bias.data();
            float* c_ptr = C.data();

            #pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    c_ptr[i * N + j] = (bias_ptr != nullptr) ? bias_ptr[j] : 0.0f;
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
                                float a_val = a_ptr[i * K + k];

                                for (int j = j0; j < j_max; ++j) {
                                    c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
                                }
                            }
                        }

                    }
                }
            }
        }
    }
}