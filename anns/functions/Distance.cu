#include <iostream>
#include <cublas_v2.h>

void computeSquaredNorms(cublasHandle_t handle, const float* matrix, int rows, int cols, float* norms) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, matrix, cols, matrix, 1, &beta, norms, 1);
}

void computeDotProduct(cublasHandle_t handle, const float* x, const float* y, int size, float* result) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSdot(handle, size, x, 1, y, 1, result);
}

void computeDistances(cublasHandle_t handle, const float* X, const float* Y, int m, int n, int k) {
    float* normsX = new float[m];
    float* normsY = new float[n];
    float* distances = new float[m * n];

    computeSquaredNorms(handle, X, m, k, normsX);
    computeSquaredNorms(handle, Y, n, k, normsY);

    float alpha = -2.0f, beta = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, X, k, Y, k, &beta, distances, m);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float dist = normsX[i] + normsY[j] + distances[i * n + j];
            std::cout << "Distance between X" << i << " and Y" << j << ": " << dist << std::endl;
        }
    }

    delete[] normsX;
    delete[] normsY;
    delete[] distances;
}

int main() {
    const int m = 3; // Number of rows in X
    const int n = 2; // Number of rows in Y
    const int k = 4; // Dimension of vectors

    float X[m * k] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0};

    float Y[n * k] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0};

    cublasHandle_t handle;
    cublasCreate(&handle);

    computeDistances(handle, X, Y, m, n, k);

    cublasDestroy(handle);
    return 0;
}