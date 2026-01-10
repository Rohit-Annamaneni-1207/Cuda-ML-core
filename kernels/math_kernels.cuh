#pragma once
#include <cuda_runtime.h>

namespace kernels {
    //y_pred = Xw
    __global__ void matvecmult(const float* X, const float*w, float* y_pred, int N, int Din, int Dout){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;

        if (idx >= Dout || idy >= N)
        {
            return;
        }

        float sum = 0;

        for (int k = 0; k<Din; k++)
        {
            float a = X[Din*idy+k];
            float b = w[Dout*k + idx];
            sum += a*b;
        }

        y_pred[Dout*idy + idx] = sum;

    }

    //grad = X.T(y_pred - y)/N
    __global__ void compute_grad(const float* X, const float*y, const float* y_pred, float* grad, int N, int Din, int Dout){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;

        if (idx >= Dout || idy >= Din)
        {
            return;
        }

        float sum = 0;

        for (int k = 0; k<N; k++)
        {
            float a = X[k*Din + idy];
            float b = y_pred[k*Dout + idx];
            float c = y[k*Dout + idx];
            sum += a*(b-c);
        }

        grad[Dout*idy + idx] = sum/N;
    }

    //w = w - lr*grad
    __global__ void update_params(float* w, const float* grad, const float lr, int Din, int Dout){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;

        if (idx >= Dout || idy >= Din)
        {
            return;
        }

        w[idy*Dout + idx] = w[idy*Dout + idx] - lr*grad[idy*Dout + idx];
    }

}