cmake_minimum_required(VERSION 3.10)

project(MyGoCppLibTorchApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)

link_directories(${LIBTORCH_PATH}/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_library(test_torch SHARED test_torch.cpp)

set_target_properties(test_torch PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(test_torch
    torch
    c10
    cudart
    cuda
    ${CUDA_LIBRARIES}
    pthread
)

set(GO_SOURCES main.go)
set(GO_LDFLAGS -L${LIBRARY_OUTPUT_DIRECTORY})

add_custom_target(
    main ALL
    COMMAND go build -o main ${GO_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(main test_torch)
