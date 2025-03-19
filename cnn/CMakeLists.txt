cmake_minimum_required(VERSION 3.10)

project(MyLibTorchApp)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_PREFIX_PATH "../py/venv/lib/python3.12/site-packages/torch")
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)

add_executable(test_cnn test_cnn.cpp)

target_link_libraries(test_cnn "${TORCH_LIBRARIES}")
set_property(TARGET test_cnn PROPERTY CXX_STANDARD 17)
