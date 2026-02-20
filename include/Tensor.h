#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mt {
    class Tensor {
    public:
        Tensor();
        explicit Tensor(const std::vector<int>& shape);
        Tensor(const std::vector<int>& shape, float* external_data);

        Tensor(const Tensor& other) = default;
        Tensor& operator=(const Tensor& other) = default;

        const std::vector<int>& shape() const { return shape_; }
        const std::vector<int>& strides() const { return strides_; }
        size_t size() const { return size_; }
        float* data() { return data_.get(); }
        const float* data() const { return data_.get(); }

        int ndims() const { return static_cast<int>(shape_.size()); }
        int dim(int index) const { return shape_.at(index); }

        // For unit tests
        float& operator()(const std::vector<int>& indices);
        float operator()(const std::vector<int>& indices) const;

        void reshape(const std::vector<int>& new_shape);

        Tensor clone() const;

        void fill(float value);

    private:
        std::vector<int> shape_;
        std::vector<int> strides_;
        size_t size_;

        std::shared_ptr<float> data_;

        void compute_strides();
    };
}