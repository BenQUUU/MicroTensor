#pragma once

#include "Tensor.h"

namespace mt {
    namespace backend {
        // C = A * B + bias (one thread version)
        void gemm_cpu_base(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C);
    }
}