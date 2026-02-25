#include "math_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 16

namespace mt {
    namespace backend {
        __global__ void gemm_tiled_kernel(const float* A, const float* B, const float* bias, float* C, int M, int K, int N) {
            int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            int col = blockIdx.x * TILE_SIZE + threadIdx.x;

            __shared__ float sA[TILE_SIZE][TILE_SIZE];
            __shared__ float sB[TILE_SIZE][TILE_SIZE];

            float sum = 0.0f;

            if (bias != nullptr && col < N && row < M) {
                sum = bias[col];
            }

            for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
                if (row < M && t * TILE_SIZE + threadIdx.x < K) {
                    sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
                } else {
                    sA[threadIdx.y][threadIdx.x] = 0.0f;
                }

                if (t * TILE_SIZE + threadIdx.y < K && col < N) {
                    sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
                } else {
                    sB[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();

                for (int i = 0; i < TILE_SIZE; ++i) {
                    sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
                }

                __syncthreads();
            }

            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }

        void launch_gemm_cuda(const float* d_A, const float* d_B, const float* d_bias, float* d_C, int M, int K, int N) {
            dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
            dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

            gemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_bias, d_C, M, K, N);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {}
        }

    }
}