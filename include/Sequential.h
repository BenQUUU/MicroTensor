#pragma once

#include "Layer.h"
#include <vector>
#include <memory>

namespace mt {
    class Sequential : public Layer {
    public:
        Sequential() = default;

        void add(std::shared_ptr<Layer> layer);

        Tensor forward(const Tensor &input) override;

    private:
        std::vector<std::shared_ptr<Layer>> layers_;
    };
}