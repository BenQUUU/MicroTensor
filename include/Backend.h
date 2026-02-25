#pragma once

#include "Tensor.h"

namespace mt {
    namespace backend {
        // C = A * B + bias (one thread version)
        void gemm_cpu_base(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);

        void gemm_cpu_tiled(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);

#ifdef USE_AVX2
        void gemm_avx2_tiled(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);
#endif
#ifdef USE_CUDA
        void gemm_cuda(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);
#endif

        void gemm(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);
    }
}