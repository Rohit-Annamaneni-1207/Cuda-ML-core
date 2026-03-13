cmake_minimum_required(VERSION 3.18)

project(CudaMLCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(cudaml STATIC
    src/linear_regression.cu
)

target_include_directories(cudaml
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaml
    PUBLIC
        CUDA::cudart
)

add_executable(debug_test
    examples/debug_test.cpp
)

target_link_libraries(debug_test
    PRIVATE
        cudaml
)

if(WIN32)
    file(GLOB CUDA_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
    foreach(cuda_dll IN LISTS CUDA_CUDART_DLLS)
        add_custom_command(TARGET debug_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${cuda_dll}"
                $<TARGET_FILE_DIR:debug_test>
        )
    endforeach()
endif()

add_executable(test_linear_regression
    examples/test_linear_regression.cpp
)

target_link_libraries(test_linear_regression
    PRIVATE
        cudaml
)

if(WIN32)
    foreach(cuda_dll IN LISTS CUDA_CUDART_DLLS)
        add_custom_command(TARGET test_linear_regression POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${cuda_dll}"
                $<TARGET_FILE_DIR:test_linear_regression>
        )
    endforeach()
endif()

enable_testing()
add_test(NAME LinearRegressionTest COMMAND test_linear_regression)
