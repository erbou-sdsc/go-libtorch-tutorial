cmake_minimum_required(VERSION 3.10)

project(MyLibTorchApp)

set(CMAKE_CXX_STANDARD 17)

file(GLOB PYTHON_TORCH_PATHS "${CMAKE_CURRENT_LIST_DIR}/../py/venv/lib/python3*/site-packages/torch/")
list(JOIN PYTHON_TORCH_PATHS ";" CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch;${PYTHON_TORCH_PATHS}")

find_package(Torch REQUIRED)

add_executable(test_cnn test_cnn.cpp)

target_link_libraries(test_cnn "${TORCH_LIBRARIES}")
set_property(TARGET test_cnn PROPERTY CXX_STANDARD 17)
