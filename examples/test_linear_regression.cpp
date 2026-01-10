#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include <cuda_runtime.h>

#include "cudaml/models/linear_regression.hpp"

// --------------------------------------------------
// Simple CPU reference implementations
// --------------------------------------------------

void cpu_forward(const std::vector<float>& X,
                 const std::vector<float>& W,
                 std::vector<float>& Y,
                 int N, int Din, int Dout)
{
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < Dout; k++) {
            float sum = 0.0f;
            for (int j = 0; j < Din; j++) {
                sum += X[i * Din + j] * W[j * Dout + k];
            }
            Y[i * Dout + k] = sum;
        }
    }
}

float cpu_mse(const std::vector<float>& Y,
              const std::vector<float>& Y_pred)
{
    assert(Y.size() == Y_pred.size());
    float sum = 0.0f;
    for (size_t i = 0; i < Y.size(); i++) {
        float diff = Y_pred[i] - Y[i];
        sum += diff * diff;
    }
    return sum / Y.size();
}

// --------------------------------------------------
// Main test
// --------------------------------------------------

int main()
{
    // Small, debuggable sizes
    const int N    = 8;
    const int Din  = 3;
    const int Dout = 2;
    const int iters = 5000;
    const float lr = 0.01f;

    std::cout << "Testing CUDA Linear Regression\n";

    // -----------------------------
    // Generate simple data
    // -----------------------------
    std::vector<float> X(N * Din);
    std::vector<float> Y(N * Dout);

    // Ground-truth weights (unknown to model)
    std::vector<float> W_true = {
        1.0f, -2.0f,
        0.5f,  1.5f,
        -1.0f, 0.3f
    };

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < Din; j++) {
            X[i * Din + j] = static_cast<float>(i + j + 1);
        }
    }

    cpu_forward(X, W_true, Y, N, Din, Dout);

    // -----------------------------
    // Allocate device memory
    // -----------------------------
    float* d_X = nullptr;
    float* d_Y = nullptr;

    cudaMalloc(&d_X, X.size() * sizeof(float));
    cudaMalloc(&d_Y, Y.size() * sizeof(float));

    cudaMemcpy(d_X, X.data(), X.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(), Y.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // -----------------------------
    // Train model on GPU
    // -----------------------------
    cudaml::LinearRegression model(Din, Dout, lr);

    model.fit(d_X, d_Y, N, iters);

    // -----------------------------
    // Predict
    // -----------------------------
    std::vector<float> Y_pred(N * Dout);
    model.predict(d_X, N, Y_pred.data());

    // -----------------------------
    // Compare results
    // -----------------------------
    float mse = cpu_mse(Y, Y_pred);

    std::cout << "MSE after training: " << mse << "\n";

    std::cout << "\nFirst few predictions:\n";
    for (int i = 0; i < std::min(N, 4); i++) {
        std::cout << "Sample " << i << ": ";
        for (int k = 0; k < Dout; k++) {
            std::cout << Y_pred[i * Dout + k] << " ";
        }
        std::cout << "\n";
    }

    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaFree(d_X);
    cudaFree(d_Y);

    std::cout << "\nTest completed.\n";
    return 0;
}
