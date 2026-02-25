#include "Backend.h"

namespace mt {
    namespace backend {
        void gemm(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
#ifdef USE_CUDA
            gemm_cuda(A, B, bias, C);
#elif defined(USE_AVX2)
            gemm_avx2_tiled(A, B, bias, C);
#elif defined(USE_NEON)
            // gemm_neon_tiled(A, B, bias, C);
            gemm_cpu_tiled(A, B, bias, C);
#else
            gemm_cpu_tiled(A, B, bias, C);
#endif
        }
    }
}