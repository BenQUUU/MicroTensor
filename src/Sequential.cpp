#include "Sequential.h"

namespace mt {
    void Sequential::add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }

    Tensor Sequential::forward(const Tensor& input) {
        Tensor current = input;

        for (const auto& layer : layers_) {
            current = layer->forward(current);
        }

        return current;
    }
}