#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <cstdlib>

// Define the CNN model
struct CNN : torch::nn::Module {
    CNN() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 1000));
        fc2 = register_module("fc2", torch::nn::Linear(1000, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
        x = x.view({-1, 64 * 7 * 7});  // Flatten the tensor
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);  // Apply softmax
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main(int argc, char const **argv) {
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

    int batch_size{64};

    if (argc > 1) {
        auto bsz = std::atoi(argv[1]);
        if (bsz < 64 || bsz > 1000) {
            std::cout << "Ignore batch size parameter " << bsz << std::endl;
        } else {
            batch_size = bsz ;
            std::cout << "Use batch size " << batch_size << std::endl;
        }
    }

    // Load the MNIST dataset, it must be saved into data
    // train-images-idx3-ubyte  train-labels-idx1-ubyte
    auto train_dataset = torch::data::datasets::MNIST("./data")
                             .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                             .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST("./data")
                            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                            .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(train_dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));
    auto test_loader = torch::data::make_data_loader(test_dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));
    auto progress_batch = int(train_dataset.size().value() / 5.0 / batch_size + 1 - 1.0/train_dataset.size().value());

    CNN model;
    model.to(device);

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    // Training
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        model.train();
        size_t batch_idx = 0;
        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto output = model.forward(data);

            auto loss = torch::nll_loss(output, target);

            loss.backward();

            optimizer.step();

            if (++batch_idx % progress_batch == 0) {
                std::cout << "Train Epoch: " << epoch << " [" << batch_idx * batch_size << "/" << train_dataset.size().value()
                          << "]\tLoss: " << loss.item<float>() << std::endl;
            }
        }
    }

    // Test
    model.eval();
    torch::Tensor correct = torch::zeros({1}, torch::kInt64).to(device);
    torch::Tensor total = torch::zeros({1}, torch::kInt64).to(device);
    for (auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model.forward(data);

        auto pred = output.argmax(1);
        correct += pred.eq(target).sum();
        total += target.size(0);
    }

    std::cout << "Test Accuracy: " << 100.0 * correct.item<int64_t>() / total.item<int64_t>() << "%" << std::endl;

    return 0;
}

