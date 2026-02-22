#pragma once

#include "Layer.h"

namespace mt {
    class Linear : public Layer {
    public:
        Linear(int in_features, int out_features);

        void load_weights(const float* w_data, const float* b_data);

        Tensor forward(const Tensor &input) override;

    private:
        Tensor weights_; // [in_features, out_features]
        Tensor bias_;    // [out_features]
    };
}