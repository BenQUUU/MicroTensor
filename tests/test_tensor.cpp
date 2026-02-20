#include <gtest/gtest.h>
#include "../include/Tensor.h"
#include <vector>

using namespace mt;

TEST(TensorTest, InitializationAndStrides) {
    Tensor t({2, 3, 4});

    EXPECT_EQ(t.ndims(), 3);
    EXPECT_EQ(t.size(), 2 * 3 * 4);

    const auto& strides = t.strides();
    ASSERT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(TensorTest, ElementAccess) {
    Tensor t({2, 2}); // 2x2

    t.data()[0] = 1.0f; // [0, 0]
    t.data()[1] = 2.0f; // [0, 1]
    t.data()[2] = 3.0f; // [1, 0]
    t.data()[3] = 4.0f; // [1, 1]

    EXPECT_FLOAT_EQ(t({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t({1, 1}), 4.0f);

    EXPECT_THROW(t({2, 0}), std::out_of_range);
    EXPECT_THROW(t({0, 0, 0}), std::invalid_argument);
}

TEST(TensorTest, Reshape) {
    Tensor t({4, 3});
    t.fill(7.0f);

    t.reshape({2, 6});
    EXPECT_EQ(t.ndims(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 6);
    EXPECT_EQ(t.size(), 12);
    EXPECT_EQ(t.strides()[0], 6);

    EXPECT_THROW(t.reshape({5, 5}), std::invalid_argument);
}

TEST(TensorTest, Clone) {
    Tensor t1({2, 2});
    t1.fill(1.0f);

    Tensor t2 = t1.clone();
    t2.fill(9.0f);

    EXPECT_FLOAT_EQ(t1({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t2({0, 0}), 9.0f);

    EXPECT_NE(t1.data(), t2.data());
}

TEST(TensorTest, ExternalMemory) {
    std::vector<float> raw_data = {1.1f, 2.2f, 3.3f, 4.4f};

    Tensor t({2, 2}, raw_data.data());

    EXPECT_FLOAT_EQ(t({1, 0}), 3.3f);

    t({0, 1}) = 9.9f;

    EXPECT_FLOAT_EQ(raw_data[1], 9.9f);
}