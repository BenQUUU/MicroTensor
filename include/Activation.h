#pragma once

#include "Layer.h"

namespace mt {
    class ReLU : public Layer {
    public:
        Tensor forward(const Tensor& input) override;
    };

    class Softmax : public Layer {
    public:
        Tensor forward(const Tensor& input) override;
    };
}