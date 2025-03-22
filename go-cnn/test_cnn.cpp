#include <torch/torch.h>
#include <exception>
#include <iostream>
#include <array>
#include <memory>
#include <numeric>
#include "test_cnn.h"

// CNN Model Abstraction
struct CNN : torch::nn::Module {
    using shape_t = at::IntArrayRef ;

    CNN(torch::Device const device) : device(device) {}
    virtual ~CNN() {}

    virtual torch::Tensor forward(torch::Tensor x) = 0 ;
    virtual const shape_t input_shape() const = 0 ;
    virtual const shape_t output_shape() const = 0 ;

    virtual const size_t input_size() const {
        return [](shape_t s) { return std::accumulate(s.begin(), s.end(), 1, std::multiplies()); }(input_shape());
    }

    virtual const size_t output_size() const {
        return [](auto s) { return std::accumulate(s.begin(), s.end(), 1, std::multiplies()); }(output_shape());
    }

    torch::Device const device;
};

// CNN Model for MNIST data
struct CNN_Mnist : CNN {
    CNN_Mnist(torch::Device const device) : CNN{device} {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 1000));
        fc2 = register_module("fc2", torch::nn::Linear(1000, 10));

        this->to(device);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
        x = x.view({-1, 64 * 7 * 7});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    const shape_t input_shape() const {
        return i_shape ;
    }

    const shape_t output_shape() const {
        return o_shape ;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    const std::array<shape_t::value_type,3> i_shape{1,28,28};
    const shape_t::value_type o_shape{10};
};

// Model Facotry
struct ModelFactory {
    using ModelConstructor = std::function<CNN*(torch::Device, std::string_view)>;

    ModelFactory() {
        registerModel("CNN_Mnist", [](torch::Device device, std::string_view opt) { return new CNN_Mnist(device); });
    }

    void registerModel(std::string&& name, ModelConstructor cnstr) {
        registry()[name] = cnstr ;
    }

    CNN* createModel(const std::string& model, torch::Device device, std::string opt) {
        auto it = registry().find(model);
        if (it != registry().end()) {
            return it->second(device, opt);
        }
        std::cerr << "Model not found - " << model << std::endl;
        return nullptr;  
    }

    static std::unordered_map<std::string, ModelConstructor>& registry() {
        static std::unordered_map<std::string, ModelConstructor> registry_;
        return registry_;
    }
};

extern "C" CNN* torch_initialize_model(char const* model, char const* options) {
    static ModelFactory factory ;

    try {
        torch::Device device{torch::kCPU};
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "CUDA is available!" << std::endl;
        } else if (torch::mps::is_available()) {
            device = torch::kMPS;
            std::cout << "MPS is available!" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }

        return factory.createModel(model, device, std::string{options});
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
	return nullptr;
    }
}

extern "C" size_t torch_model_output_size(CNN const* model) {
    return model->output_size();
}

extern "C" size_t torch_model_input_size(CNN const* model) {
    return model->input_size();
}

// Function to delete the model
extern "C" void torch_delete_model(CNN* model) {
    delete model;
}


// Initialize model on the correct device
extern "C" void torch_training(CNN* model, float* data, int * target, size_t data_size, int num_epochs) {
    try {
	size_t batch_size = data_size / model->input_size() ;
	if (batch_size * model->input_size() != data_size) {
            std::cerr << "Input size is not a multiple of " << model->input_size() << std::endl;
	    return ;
	}

	std::vector<at::IntArrayRef::value_type> shape;
	shape.push_back(static_cast<at::IntArrayRef::value_type>(batch_size));
	shape.insert(shape.end(), model->input_shape().begin(), model->input_shape().end());

        torch::Tensor data_tensor = torch::from_blob(data, shape, torch::kFloat).to(model->device);
        torch::Tensor target_tensor = torch::from_blob(target, {static_cast<long>(batch_size)}, torch::kInt64).to(model->device);
        //std::cout << "Data   tensor, on device: " << data_tensor.device() << ", shape: " << data_tensor.sizes() << std::endl;
        //std::cout << "Target tensor, on device: " << target_tensor.device() << ", shape: " << data_tensor.sizes() << std::endl;

        model->train();

        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            optimizer.zero_grad();
            auto output = model->forward(data_tensor);
            auto loss = torch::nll_loss(output, target_tensor);
    
            loss.backward();
    
            optimizer.step();
    
            //std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] Loss: " << loss.item<float>() << std::endl;
        }
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
    }
}

// Inference function
// you must specify a result_buffer or buffer_size large enough for the results, the function returns the
// effective size of the data saved into the buffer.
extern "C" size_t torch_inference(CNN* model, float* data, size_t data_size, float* result_buffer, size_t buffer_size) {
    try {
        c10::InferenceMode guard;
	size_t batch_size = data_size / model->input_size() ;
	if (batch_size * model->input_size() != data_size) {
            std::cerr << "Input size is not a multiple of " << model->input_size() << std::endl;
	    return 0;
	}

	std::vector<at::IntArrayRef::value_type> shape;
	shape.push_back(static_cast<at::IntArrayRef::value_type>(batch_size));
	shape.insert(shape.end(), model->input_shape().begin(), model->input_shape().end());

        torch::Tensor data_tensor = torch::from_blob(data, shape, torch::kFloat).to(model->device);
        model->eval();
        auto output = model->forward(data_tensor);

        auto output_flat = output.view({-1});

        size_t result_size = output_flat.size(0);

        if (result_size <= buffer_size) {
            auto output_cpu = output_flat.to(torch::kCPU);
            std::memcpy(result_buffer, output_cpu.data_ptr<float>(), result_size * sizeof(float));
        } else {
	    std::cerr << "Insufficient buffer size " << buffer_size << ", " << result_size << " is needed" << std::endl ;
            return 0;
        }

        return result_size;
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
	return 0;
    }
}


