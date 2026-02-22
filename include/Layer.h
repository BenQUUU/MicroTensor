#pragma once

#include "Tensor.h"

namespace mt {
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor& input) = 0;
    };
}