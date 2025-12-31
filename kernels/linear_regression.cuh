#pragma once
#include <cuda_runtime.h>

namespace cudaml::kernels {
    //y_pred = Xw
    __global__ void matvecmult(const float* X, const float*w, float* y_pred, int N, int Din, int Dout);

    //grad = X.T(y_pred - y)
    __global__ void compute_grad(const float* X, const float*y, const float* y_pred, float* grad, int N, int Din, int Dout);

    //w = w - lr*grad
    __global__ void update_params(float* w, const float* grad, float lr, int Din, int Dout);

}