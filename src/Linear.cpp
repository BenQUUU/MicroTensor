#include "../include/Linear.h"
#include "Backend.h"
#include <stdexcept>
#include <algorithm>

namespace mt {
    Linear::Linear(int in_features, int out_features)
    : weights_({in_features, out_features}), bias_({out_features}) {
        // Initializes to zeros by default (although in a real network these would be random values)
        weights_.fill(0.0f);
        bias_.fill(0.0f);
    }

    void Linear::load_weights(const float* w_data, const float* b_data) {
        if (w_data) {
            std::copy(w_data, w_data + weights_.size(), weights_.data());
        }
        if (b_data) {
            std::copy(b_data, b_data + bias_.size(), bias_.data());
        }
    }

    Tensor Linear::forward(const Tensor& input) {
        // 2D Input: [Batch_size, in_features]
        if (input.ndims() != 2) {
            throw std::invalid_argument("Linear layer expects 2D input tensor [batch, in_features]");
        }
        if (input.dim(1) != weights_.dim(0)) {
            throw std::invalid_argument("Dimension mismatch. input.features must equal weights.in_features");
        }

        int batch_size = input.dim(0);
        int out_features = weights_.dim(1);

        // [Batch_size, out_features]
        Tensor output({batch_size, out_features});

        backend::gemm_cpu_base(input, weights_, bias_, output);

        return output;
    }
}