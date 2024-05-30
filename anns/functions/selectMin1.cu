#include <iostream>
#include <vector>
#include "selectMin1.cuh"

// CUDA kernel函数，用于在每个段中查找最小值的索引
__global__ void findMinIndicesKernel(const float* values, int c, int num, int* minIndices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        int startIdx = idx * c;
        int endIdx = min((idx + 1) * c, num * c);
        float minVal = values[startIdx];
        int minIdx = 0;
        for (int i = startIdx + 1; i < endIdx; ++i) {
            if (values[i] < minVal) {
                minVal = values[i];
                minIdx = i - startIdx;
            }
        }
        minIndices[idx] = minIdx;
    }
}

// 主机函数，调用CUDA kernel函数并将结果复制回主机内存
std::vector<int> findMinIndicesCUDA(const float* values, int c, int num) {
    int totalNum = num;
    std::vector<int> minIndices(totalNum, 0); // 初始化为0

    float* d_values;
    int* d_minIndices;
    cudaMalloc((void**)&d_values, sizeof(float) * c * num);
    cudaMalloc((void**)&d_minIndices, sizeof(int) * totalNum);

    cudaMemcpy(d_values, values, sizeof(float) * c * num, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (totalNum + blockSize - 1) / blockSize;

    findMinIndicesKernel<<<numBlocks, blockSize>>>(d_values, c, num, d_minIndices);

    cudaMemcpy(minIndices.data(), d_minIndices, sizeof(int) * totalNum, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_minIndices);

    return minIndices;
}

// int main() {
//     // 示例数据
//     std::vector<float> values = {1.2, 2.4, 3.1, 0.5, 2.0, 1.8, 3.5, 4.2, 0.9, 2.3, 235, 2.4};

//     // 每段的大小和段数
//     int c = 3;
//     int num = 4;

//     // 使用CUDA加速查找每段中的最小值索引
//     std::vector<int> minIndices = findMinIndicesCUDA(values.data(), c, num);

//     // 打印结果
//     std::cout << "Minimum indices in each segment:" << std::endl;
//     for (int i = 0; i < num; ++i) {
//         std::cout << "Segment " << i << ": " << minIndices[i] << std::endl;
//     }

//     return 0;
// }