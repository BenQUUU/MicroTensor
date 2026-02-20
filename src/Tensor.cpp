#include "../include/Tensor.h"
#include <numeric>
#include <algorithm>

namespace mt {
    void Tensor::compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;

        int stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    Tensor::Tensor() : size_(0) {}

    Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
        size_ = 1;
        for (int dim : shape) {
            if (dim <= 0) throw std::invalid_argument("The dimensions of the tensor must be positive");
            size_ *= dim;
        }

        compute_strides();

        data_ = std::shared_ptr<float>(new float[size_](), [](float* ptr) { delete[] ptr; });
    }

    Tensor::Tensor(const std::vector<int>& shape, float* external_data) : shape_(shape) {
        size_ = 1;
        for (int dim : shape) {
            if (dim <= 0) throw std::invalid_argument("The dimensions of the tensor must be positive");
            size_ *= dim;
        }

        compute_strides();

        data_ = std::shared_ptr<float>(external_data, [](float*) {});
    }

    float& Tensor::operator()(const std::vector<int>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Wrong number of dimensions in index");
        }
        int flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            flat_index += indices[i] * strides_[i];
        }
        return data_.get()[flat_index];
    }

    float Tensor::operator()(const std::vector<int>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Wrong number of dimensions in index");
        }
        int flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            flat_index += indices[i] * strides_[i];
        }
        return data_.get()[flat_index];
    }

    void Tensor::reshape(const std::vector<int>& new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) new_size *= dim;

        if (new_size != size_) {
            throw std::invalid_argument("The new shape must have exactly the same number of elements (Volume)");
        }

        shape_ = new_shape;
        compute_strides();
    }

    Tensor Tensor::clone() const {
        Tensor copy(shape_);
        std::copy(data_.get(), data_.get() + size_, copy.data_.get());
        return copy;
    }

    void Tensor::fill(float value) {
        std::fill(data_.get(), data_.get() + size_, value);
    }
}
