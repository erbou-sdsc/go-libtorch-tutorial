#include <torch/torch.h>
#include <iostream>
#include "test_sum.h"

float _test_sum(int num, const float* array) {
    try {
        torch::Device device(torch::kMPS);
    
        if (!torch::cuda::is_available() && !device.is_mps()) {
            std::cerr << "MPS is not available on this device!" << std::endl;
            // torch::Device device(torch::kCPU);
            return -1;
        }
    
        torch::Tensor tensor = torch::from_blob(const_cast<float*>(array), {num}, torch::kFloat).to(device);
        torch::Tensor sum = tensor.sum();

        return sum.item<float>();
    } catch (...) {
        return 0.0;
    }
}

extern "C" {
    float test_sum(int num, const float* array) {
        return _test_sum(num, array);
    }
}

