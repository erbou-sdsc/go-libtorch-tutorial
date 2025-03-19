cmake_minimum_required(VERSION 3.10)

project(MyGoCppLibTorchApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB PYTHON_TORCH_PATHS "${CMAKE_CURRENT_LIST_DIR}/../py/venv/lib/python3*/site-packages/torch/")
list(JOIN PYTHON_TORCH_PATHS ";" CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch;${PYTHON_TORCH_PATHS}")

find_package(Torch REQUIRED)

find_package(CUDA QUIET)

if(CUDA_FOUND)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_LIBRARIES cuda cudart ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA not found, skipping CUDA")
    set(CUDA_LIBRARIES "")
endif()

link_directories(${LIBTORCH_PATH}/lib)
add_library(test_torch SHARED test_torch.cpp)

set_target_properties(test_torch PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(test_torch
    torch
    c10
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
