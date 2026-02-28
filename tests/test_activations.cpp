#include <gtest/gtest.h>
#include "Activation.h"
#include "Sequential.h"
#include "Linear.h"

using namespace mt;

TEST(ActivationTest, ReLU_Logic) {
    Tensor input({2, 2});
    input({0, 0}) = -1.5f; 
    input({0, 1}) = 3.1f;
    input({1, 0}) = -100.0f; 
    input({1, 1}) = 0.0f;

    ReLU relu;
    Tensor output = relu.forward(input);

    EXPECT_FLOAT_EQ(output({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output({0, 1}), 3.1f);
    EXPECT_FLOAT_EQ(output({1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output({1, 1}), 0.0f);
}

TEST(ActivationTest, Softmax_Distribution) {
    Tensor input({1, 3});
    input({0, 0}) = 1.0f; 
    input({0, 1}) = 2.0f; 
    input({0, 2}) = 3.0f;

    Softmax softmax;
    Tensor output = softmax.forward(input);

    float sum = output({0, 0}) + output({0, 1}) + output({0, 2});
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    EXPECT_GT(output({0, 2}), output({0, 1}));
    EXPECT_GT(output({0, 1}), output({0, 0}));
}

TEST(SequentialTest, ForwardFlow) {
    Sequential model;

    auto fc1 = std::make_shared<Linear>(3, 2);
    model.add(fc1);

    model.add(std::make_shared<ReLU>());

    Tensor input({1, 3});
    input.fill(1.0f);

    Tensor output = model.forward(input);

    EXPECT_EQ(output.ndims(), 2);
    EXPECT_EQ(output.dim(0), 1);
    EXPECT_EQ(output.dim(1), 2);
}