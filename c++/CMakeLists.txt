cmake_minimum_required(VERSION 3.10)

project(MyLibTorchApp)

set(CMAKE_CXX_STANDARD 17)

set(NVCC_APPEND_FLAGS "${NVCC_APPEND_FLAGS} -allow-unsupported-compile")

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)

add_executable(test_torch test_torch.cpp)

target_link_libraries(test_torch "${TORCH_LIBRARIES}")
set_property(TARGET test_torch PROPERTY CXX_STANDARD 17)
