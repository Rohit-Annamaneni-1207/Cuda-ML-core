#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#include "cudaml/models/linear_regression.hpp"
#include "kernels/linear_regression.cuh"

namespace cudaml {
    LinearRegression::LinearRegression(int input_features_num, int output_features_num, float lr): Din(input_features_num), Dout(output_features_num), lr(lr), d_w(nullptr){
        cudaError_t err = cudaMalloc(&d_w, Din*Dout*sizeof(float));
        if (err != cudaSucess)
        {
            throw std::runtime_error("cudaMalloc failed to allocate memory for weights");
        }
        
        //intitialize weights
        cudaMemset(d_w, 0, Din*Dout*sizeof(float));
    }

    LinearRegression::~LinearRegression()
    {
        if (d_w)
        {
            cudaFree(d_w);
        }
    }
}

