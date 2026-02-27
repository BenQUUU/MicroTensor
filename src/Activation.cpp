#include "Activation.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mt {
    Tensor ReLU::forward(const Tensor& input) {
        Tensor output = input.clone();
        float* data = output.data();
        long long total_size = static_cast<long long>(output.size());

        #pragma omp parallel for
        for (long long i = 0; i < total_size; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }

        return output;
    }

    Tensor Softmax::forward(const Tensor& input) {
        if (input.ndims() != 2) {
            throw std::invalid_argument("Softmax expects 2D tensor [Batch, Features]");
        }

        Tensor output = input.clone();
        int batch_size = output.dim(0);
        int features = output.dim(1);
        float* data = output.data();

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            int offset = b * features;

            float max_val = data[offset];
            for (int f = 1; f < features; ++f) {
                max_val = std::max(max_val, data[offset + f]);
            }

            float sum_exp = 0.0f;
            for (int f = 0; f < features; ++f) {
                data[offset + f] = std::exp(data[offset + f] - max_val);
                sum_exp += data[offset + f];
            }

            for (int f = 0; f < features; ++f) {
                data[offset + f] /= sum_exp;
            }
        }

        return output;
    }
}