#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Device device(torch::kMPS);

    if (!torch::cuda::is_available() && !device.is_mps()) {
        //  Use MPS if available on the device
        std::cerr << "MPS is not available on this device!" << std::endl;
        // torch::Device device(torch::kCPU);
        return -1;
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
