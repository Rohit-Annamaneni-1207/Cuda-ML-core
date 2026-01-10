#pragma once

namespace cudaml{

    class LinearRegression{
        public:
            LinearRegression(int input_features_num, int output_features_num, float lr);
            ~LinearRegression();

            void fit(const float* d_X, const float* d_y, int num_samples, int num_iters);
            void predict(const float* d_X, int num_samples, float* y_pred);
            const float* weights();

        private:
            int Din; //number of features input
            int Dout; //number of features ouput
            float lr; // learning rate
            float* d_w; // pointer to weights on device
    };
}