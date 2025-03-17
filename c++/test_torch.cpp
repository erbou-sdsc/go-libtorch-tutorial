#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Device device(torch::kCPU);

    if (torch::mps::is_available()) {
        torch::Device device(torch::kMPS);
	if (!device.is_mps()) {
            std::cout << "MPS is not available on this device!" << std::endl;
	} else {
            std::cout << "MPS is available on this device!" << std::endl;
	}
    } else if (torch::cuda::is_available()) {
	torch::Device device(torch::kCUDA);
	if (!device.is_cuda()) {
            std::cout << "CUDA is not available on this device!" << std::endl;
	} else {
            std::cout << "CUDA is available on this device!" << std::endl;
	}
    } else {
            std::cout << "Using CPU" << std::endl;
    }


    // Create a 6x6 tensor
    at::Tensor tensor = torch::rand({6, 6}, device);
    std::cout << "Original Tensor:" << std::endl;
    std::cout << tensor << std::endl;

    // Perform the sum operation
    at::Tensor sum = tensor.sum();  // Summing all the elements
    std::cout << "Sum of all elements:" << std::endl;
    std::cout << sum.item<float>() << std::endl;

    // Sum along a particular dimension (e.g., sum each column),
    // Tells us about row / column ordering.
    std::cout << "Sum along rows:" << std::endl;
    at::Tensor sum_dim0 = tensor.sum(0);
    std::cout << "sum along cols (dim 0):\n" << sum_dim0 << std::endl;
    std::cout << "sum along rows (dim 1):\n" << tensor.sum(1) << std::endl;
    std::cout << "sum along rows + cols:\n" << tensor.sum(1).sum(0) << std::endl;

    return 0;
}
