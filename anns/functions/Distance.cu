#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath> 

void printMatrix(const float* mat, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(2) << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

__global__ void computeDistancesKernel(float* X, float* Y, float* D, int m, int n, int k) {
    // Each thread computes one element of the distance matrix D
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Row index in X
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Column index in Y

    if (idx < m && idy < n) {
        float sum = 0.0f;
        // Compute dot product
        for (int i = 0; i < k; i++) {
            sum += X[idx * k + i] * Y[idy * k + i];
        }
        // Compute the distance squared
        float normX = 0.0f, normY = 0.0f;
        for (int i = 0; i < k; i++) {
            normX += X[idx * k + i] * X[idx * k + i];
            normY += Y[idy * k + i] * Y[idy * k + i];
        }
        D[idx * n + idy] = normX + normY - 2 * sum;
    }
}

void computeDistances(float* X, float* Y, float* D, int m, int n, int k) {
    float *dX, *dY, *dD;
    size_t sizeX = m * k * sizeof(float);
    size_t sizeY = n * k * sizeof(float);
    size_t sizeD = m * n * sizeof(float);

    cudaMalloc(&dX, sizeX);
    cudaMalloc(&dY, sizeY);
    cudaMalloc(&dD, sizeD);

    cudaMemcpy(dX, X, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, sizeY, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); // Choose reasonable block sizes
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    computeDistancesKernel<<<gridSize, blockSize>>>(dX, dY, dD, m, n, k);

    cudaMemcpy(D, dD, sizeD, cudaMemcpyDeviceToHost);

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
}

int main() {
    const int m = 1024, n = 1024, k = 128; // Example sizes
    float *X = new float[m * k];
    float *Y = new float[n * k];
    float *D = new float[m * n];

    // Initialize X and Y with random data for demonstration purposes
    for (int i = 0; i < m * k; i++) X[i] = static_cast<float>(1);
    for (int i = 0; i < n * k; i++) Y[i] = static_cast<float>(0);

    computeDistances(X, Y, D, m, n, k);

    // Print X, Y, and D matrices
    printMatrix(X, m, k, "X");
    printMatrix(Y, n, k, "Y");
    printMatrix(D, m, n, "Distance Squared");

    // Free memory
    delete[] X;
    delete[] Y;
    delete[] D;

    return 0;
}