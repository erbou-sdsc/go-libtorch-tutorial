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


    // Create a tensor
    at::Tensor tensor = torch::rand({3, 3}, device);  // 3x3 tensor with random values
    std::cout << "Original Tensor:" << std::endl;
    std::cout << tensor << std::endl;

    // Perform the sum operation
    at::Tensor sum = tensor.sum();  // Summing all the elements
    std::cout << "Sum of all elements:" << std::endl;
    std::cout << sum.item<float>() << std::endl;  // Convert the sum to a float for display

    // Optionally, sum along a particular dimension (e.g., sum each column)
    at::Tensor sum_dim0 = tensor.sum(0);  // Sum along the 0th dimension (rows)
    std::cout << "Sum along rows (dim 0):" << std::endl;
    std::cout << sum_dim0 << std::endl;

    return 0;
}
