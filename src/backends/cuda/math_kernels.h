#pragma once

namespace mt {
    namespace backend {
        void launch_gemm_cuda(const float* d_A, const float* d_B, const float* d_bias, float* d_C,
                              int M, int K, int N);

    }
}