#include "Backend.h"

#ifdef USE_CUDA
#include "math_kernels.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace mt {
    namespace backend {
        void gemm_cuda(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
            int M = A.dim(0);
            int K = A.dim(1);
            int N = B.dim(1);

            size_t size_A = M * K * sizeof(float);
            size_t size_B = K * N * sizeof(float);
            size_t size_C = M * N * sizeof(float);
            size_t size_bias = (bias.size() > 0) ? N * sizeof(float) : 0;

            float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_bias = nullptr;

            try {
                if (cudaMalloc(&d_A, size_A) != cudaSuccess) throw std::runtime_error("CUDA malloc failed for A");
                if (cudaMalloc(&d_B, size_B) != cudaSuccess) throw std::runtime_error("CUDA malloc failed for B");
                if (cudaMalloc(&d_C, size_C) != cudaSuccess) throw std::runtime_error("CUDA malloc failed for C");

                if (size_bias > 0) {
                    if (cudaMalloc(&d_bias, size_bias) != cudaSuccess) throw std::runtime_error("CUDA malloc failed for bias");
                    cudaMemcpy(d_bias, bias.data(), size_bias, cudaMemcpyHostToDevice);
                }

                cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);

                launch_gemm_cuda(d_A, d_B, d_bias, d_C, M, K, N);

                cudaDeviceSynchronize();
                cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                if (d_bias) cudaFree(d_bias);

            } catch (const std::exception& e) {
                if (d_A) cudaFree(d_A);
                if (d_B) cudaFree(d_B);
                if (d_C) cudaFree(d_C);
                if (d_bias) cudaFree(d_bias);
                throw;
            }
        }
    }
}
#endif