#pragma once

// #include <cuda.h>

__global__ void findMinIndicesKernel(const float* values, int c, int num, int* minIndices);

std::vector<int> findMinIndicesCUDA(const float* values, int c, int num)
