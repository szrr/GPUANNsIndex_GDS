#pragma once

// #include <cuda.h>

__global__ void findMinIndicesKernel(const float* values, int c, int num, int* minIndices);

std::vector<int> findMinIndices(const float* values, int c, int num);

void deviceFindMinIndices(const float* values, int c, int num, int* d_min_indices);