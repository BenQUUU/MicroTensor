#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>

#include "Tensor.h"
#include "Linear.h"
#include "Activation.h"
#include "Sequential.h"

using namespace mt;

std::vector<float> read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Critical error: Cannot open file " + filepath);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate a vector with a size corresponding to the number of floats (bytes / 4)
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Critical error: Failed to read file " + filepath);
    }
    return buffer;
}

int main() {
    try {
        std::cout << "--- MicroTensor Initialization ---" << std::endl;

        std::cout << "[1/4] Loading data from disk..." << std::endl;
        std::vector<float> weights_data = read_binary_file("mnist_weights.bin");
        std::vector<float> image_data = read_binary_file("test_image.bin");

        std::cout << "[2/4] Constructing model architecture..." << std::endl;
        Sequential model;

        auto fc1 = std::make_shared<Linear>(784, 128);
        auto relu = std::make_shared<ReLU>();
        auto fc2 = std::make_shared<Linear>(128, 10);
        auto softmax = std::make_shared<Softmax>();

        int w1_size = 784 * 128;
        int b1_size = 128;
        int w2_size = 128 * 10;
        int b2_size = 10;

        if (weights_data.size() != (w1_size + b1_size + w2_size + b2_size)) {
            throw std::runtime_error("Mismatch between weights file size and expected architecture.");
        }

        const float* ptr = weights_data.data();
        fc1->load_weights(ptr, ptr + w1_size);
        ptr += w1_size + b1_size;
        fc2->load_weights(ptr, ptr + w2_size);

        model.add(fc1);
        model.add(relu);
        model.add(fc2);
        model.add(softmax);

        std::cout << "[3/4] Running inference..." << std::endl;

        Tensor input({1, 784}, image_data.data());

        auto start = std::chrono::high_resolution_clock::now();
        Tensor output = model.forward(input);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> inference_time = end - start;

        std::cout << "[4/4] Results:" << std::endl;

        int best_class = 0;
        float best_prob = output({0, 0});

        for (int i = 1; i < 10; ++i) {
            float prob = output({0, i});
            if (prob > best_prob) {
                best_prob = prob;
                best_class = i;
            }
        }

        std::cout << "=========================================" << std::endl;
        std::cout << "Recognized digit  : " << best_class << std::endl;
        std::cout << "Model confidence  : " << (best_prob * 100.0f) << "%" << std::endl;
        std::cout << "Inference time    : " << inference_time.count() << " ms" << std::endl;
        std::cout << "=========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
