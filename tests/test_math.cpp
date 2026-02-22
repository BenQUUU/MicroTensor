#include <gtest/gtest.h>
#include "Linear.h"
#include <vector>

using namespace mt;

TEST(MathTest, LinearForwardPass) {
    Linear layer(3, 2);

    std::vector<float> w_data = {
        7.0f,  8.0f,
        9.0f,  10.0f,
        11.0f, 12.0f
    };
    std::vector<float> b_data = {1.0f, 2.0f};

    layer.load_weights(w_data.data(), b_data.data());

    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    Tensor input({2, 3}, input_data.data());

    // Forward Pass
    Tensor output = layer.forward(input);

    // [Batch, Out_Features] -> [2, 2])
    ASSERT_EQ(output.ndims(), 2);
    EXPECT_EQ(output.dim(0), 2);
    EXPECT_EQ(output.dim(1), 2);

    EXPECT_FLOAT_EQ(output({0, 0}), 59.0f);
    EXPECT_FLOAT_EQ(output({0, 1}), 66.0f);
    EXPECT_FLOAT_EQ(output({1, 0}), 140.0f);
    EXPECT_FLOAT_EQ(output({1, 1}), 156.0f);
}

TEST(MathTest, LinearDimensionMismatch) {
    Linear layer(5, 2);
    
    std::vector<float> bad_input_data = {1.0f, 2.0f, 3.0f};
    Tensor bad_input({1, 3}, bad_input_data.data());

    EXPECT_THROW(layer.forward(bad_input), std::invalid_argument);
}