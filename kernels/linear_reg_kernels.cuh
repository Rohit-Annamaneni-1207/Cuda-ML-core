#pragma once
#include <cuda_runtime.h>

#define TILE_WIDTH_M 2
#define TILE_WIDTH_N 2
#define TILE_WIDTH_K 2

namespace kernels {
    //y_pred = Xw
    __global__ void tiled_matmul(const float* X, const float*w, float* y_pred, int N, int Din, int Dout)
    {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * TILE_WIDTH_M + ty;  // sample index (0..N-1)
        int col = bx * TILE_WIDTH_N + tx;  // output feature index (0..Dout-1)

        __shared__ float Xs[TILE_WIDTH_M][TILE_WIDTH_K];
        __shared__ float ws[TILE_WIDTH_K][TILE_WIDTH_N];

        float sum = 0.0f;

        // Iterate over tiles in the Din dimension
        int numTiles = (Din + TILE_WIDTH_K - 1) / TILE_WIDTH_K;
        for (int phase = 0; phase < numTiles; ++phase)
        {
            // Load tile of X into shared memory
            // Xs[ty][tx] corresponds to X[row, phase*TILE_WIDTH_K + tx]
            int x_col = phase * TILE_WIDTH_K + tx;
            if (row < N && x_col < Din)
            {
                Xs[ty][tx] = X[row * Din + x_col];
            }
            else
            {
                Xs[ty][tx] = 0.0f;
            }

            // Load tile of w into shared memory
            // ws[ty][tx] corresponds to w[phase*TILE_WIDTH_K + ty, col]
            int w_row = phase * TILE_WIDTH_K + ty;
            if (w_row < Din && col < Dout)
            {
                ws[ty][tx] = w[w_row * Dout + col];
            }
            else
            {
                ws[ty][tx] = 0.0f;
            }

            __syncthreads();

            // Compute partial product for this tile
            for (int k = 0; k < TILE_WIDTH_K; ++k)
            {
                sum += Xs[ty][k] * ws[k][tx];
            }

            __syncthreads();
        }

        // Write result to global memory
        if (row < N && col < Dout)
        {
            y_pred[row * Dout + col] = sum;
        }
    }

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