#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#include "cudaml/models/linear_regression.hpp"
#include "kernels/linear_reg_kernels.cuh"


//to define checking of silent cuda errors instead of writing code at every call
//inline functions are essentially a request to just put the function code at the point where it is invoked instead of creating overhead by actually calling the function with a stack
inline void cuda_check(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}


namespace cudaml {
    LinearRegression::LinearRegression(int input_features_num, int output_features_num, float lr, int threads_per_block): Din(input_features_num), Dout(output_features_num), lr(lr), d_w(nullptr), threadsPerBlock(threads_per_block){
        //allocate device memory on GPU
        cuda_check(cudaMalloc(&d_w, Din*Dout*sizeof(float)));
        
        //intitialize weights
        cuda_check(cudaMemset(d_w, 0, Din*Dout*sizeof(float)));
    }

    LinearRegression::~LinearRegression()
    {
        if (d_w)
        {
            //Free device memory to prevent memory leak
            cuda_check(cudaFree(d_w));
        }
    }

    void LinearRegression::fit(const float* d_X, const float* d_y, int num_samples, int num_iters)
    {
        float* d_y_pred = nullptr;
        float* d_grad = nullptr;
        int pred_size = num_samples*Dout*sizeof(float);
        int grad_size = Din*Dout*sizeof(float);

        cuda_check(cudaMalloc(&d_y_pred, pred_size));
        cuda_check(cudaMalloc(&d_grad, grad_size));

        for (int T = 0; T < num_iters; T++)
        {
            //Call kernels to predict, compute gradient and update the weights on device memory
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid(
                (Dout + threadsPerBlock.x - 1)/threadsPerBlock.x,
                (num_samples + threadsPerBlock.y - 1)/threadsPerBlock.y
            );


            kernels::matvecmult<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_w, d_y_pred, num_samples, Din, Dout);

            blocksPerGrid.x = (Dout + threadsPerBlock.x-1)/threadsPerBlock.x;
            blocksPerGrid.y = (Din + threadsPerBlock.y-1)/threadsPerBlock.y;
            cuda_check(cudaMemset(d_grad, 0, Din * Dout * sizeof(float)));
            kernels::compute_grad<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_y, d_y_pred, d_grad, num_samples, Din, Dout);
            kernels::update_params<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_grad, lr, Din, Dout);

            cuda_check(cudaDeviceSynchronize());
        }

        //No longer need preds and gradients on device memory
        cuda_check(cudaFree(d_y_pred));
        cuda_check(cudaFree(d_grad));
    }

    void LinearRegression::predict(const float* d_X, int num_samples, float* y_pred){
        float* d_y_pred = nullptr;
        int pred_size = num_samples*Dout*sizeof(float);
        cuda_check(cudaMalloc(&d_y_pred, pred_size));
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid(
            (Dout + threadsPerBlock.x - 1)/threadsPerBlock.x,
            (num_samples + threadsPerBlock.y - 1)/threadsPerBlock.y
        );


        kernels::matvecmult<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_w, d_y_pred, num_samples, Din, Dout);


        //sync before cpu can access device memory
        cuda_check(cudaDeviceSynchronize());

        // cuda memory copy to y_pred
        cuda_check(cudaMemcpy(
            y_pred,
            d_y_pred,
            pred_size,
            cudaMemcpyDeviceToHost
        ));

        //free preds on device memory
        cuda_check(cudaFree(d_y_pred));
    }

    const float* LinearRegression::weights(){
        return d_w;
    }
}

