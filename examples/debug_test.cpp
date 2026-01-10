#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cudaml/models/linear_regression.hpp"

int main() {
    const int N = 4;
    const int Din = 2;
    const int Dout = 1;
    
    // Simple data: y = 2*x1 + 3*x2
    std::vector<float> X = {
        1.0f, 1.0f,
        2.0f, 2.0f,
        3.0f, 3.0f,
        4.0f, 4.0f
    };
    
    std::vector<float> Y = {
        5.0f,   // 2*1 + 3*1
        10.0f,  // 2*2 + 3*2
        15.0f,  // 2*3 + 3*3
        20.0f   // 2*4 + 3*4
    };
    
    float* d_X = nullptr;
    float* d_Y = nullptr;
    
    cudaMalloc(&d_X, X.size() * sizeof(float));
    cudaMalloc(&d_Y, Y.size() * sizeof(float));
    
    cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "Training...\n";
    cudaml::LinearRegression model(Din, Dout, 0.01f);
    
    // Print initial weights
    std::vector<float> w_init(Din * Dout);
    cudaMemcpy(w_init.data(), model.weights(), Din * Dout * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Initial weights: ";
    for (auto w : w_init) std::cout << w << " ";
    std::cout << "\n";
    
    model.fit(d_X, d_Y, N, 10);
    
    // Print final weights
    std::vector<float> w_final(Din * Dout);
    cudaMemcpy(w_final.data(), model.weights(), Din * Dout * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Final weights: ";
    for (auto w : w_final) std::cout << w << " ";
    std::cout << "\n";
    
    // Predict
    std::vector<float> Y_pred(N * Dout);
    model.predict(d_X, N, Y_pred.data());
    
    std::cout << "Predictions vs Actual:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  " << Y_pred[i] << " vs " << Y[i] << "\n";
    }
    
    cudaFree(d_X);
    cudaFree(d_Y);
    
    return 0;
}
