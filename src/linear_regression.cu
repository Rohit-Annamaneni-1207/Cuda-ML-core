#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#include "cudaml/models/linear_regression.hpp"
#include "kernels/math_kernels.cuh"

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

    void LinearRegression::fit(const float* d_X, const float* d_y, int num_samples, int num_iters)
    {
        float* d_y_pred = nullptr;
        float* d_grad = nullptr;
        int pred_size = num_samples*Dout*sizeof(float);
        int grad_size = Din*Dout*sizeof(float);

        cudaMalloc(&d_y_pred, pred_size);
        cudaMalloc(&d_grad, grad_size);

        for (int T = 0; T < num_iters; T++)
        {
            //Call kernels and copy back to MM
        }
        //free cuda mem
    }
}

