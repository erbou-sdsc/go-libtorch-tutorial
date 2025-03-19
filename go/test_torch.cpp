#include <torch/torch.h>
#include <iostream>
#include <exception>
#include "test_torch.h"

float _test_sum(int num, const float* array) {
    try {
        torch::Device device(torch::kCPU);
        if (torch::mps::is_available()) {
            device = torch::Device(torch::kMPS);
	    if (!device.is_mps()) {
                std::cout << "MPS is not available on this device!" << std::endl;
	    } else {
                std::cout << "MPS is available on this device!" << std::endl;
	    }
        } else if (torch::cuda::is_available()) {
	    device = torch::Device(torch::kCUDA);
	    if (!device.is_cuda()) {
                std::cout << "CUDA is not available on this device!" << std::endl;
	    } else {
                std::cout << "CUDA is available on this device!" << std::endl;
	    }
        } else {
            std::cout << "Using CPU" << std::endl;
        }

        torch::Tensor tensor = torch::from_blob(const_cast<float*>(array), {num}, torch::kFloat).to(device);
        torch::Tensor sum = tensor.sum();

        return sum.item<float>();
    } catch (const std::exception& e) {
	// You can't propagate the exception to go,
	std::cerr << "Caught " << e.what() << std::endl ;
	return 0.0;
    }
}

extern "C" {
    float test_sum(int num, const float* array) {
        return _test_sum(num, array);
    }
}

