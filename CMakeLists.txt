cmake_minimum_required(VERSION 3.18)
project(CudaML LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Library sources
set(LIBRARY_SOURCES
    src/linear_regression.cu
)

# Create library
add_library(cudaml STATIC ${LIBRARY_SOURCES})
target_include_directories(cudaml PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(cudaml PUBLIC CUDA::cudart)

# Set CUDA flags
target_compile_options(cudaml PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Examples
add_executable(test_linear_regression examples/test_linear_regression.cpp)
target_link_libraries(test_linear_regression PRIVATE cudaml)

add_executable(debug_test examples/debug_test.cpp)
target_link_libraries(debug_test PRIVATE cudaml)

# Enable testing
enable_testing()
add_test(NAME LinearRegressionTest COMMAND test_linear_regression)
