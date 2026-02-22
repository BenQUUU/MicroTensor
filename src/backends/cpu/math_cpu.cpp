#include "Backend.h"

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
    }
}