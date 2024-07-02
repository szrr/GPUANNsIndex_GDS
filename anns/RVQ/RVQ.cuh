/**
 * @author szr
 * @date 2024/5/24
 * @brief two-layer k-means using RVQ method
 * 
 * **/

#pragma once
#include <iostream>
#include <vector>
// #include <mkl_cblas.h>
// #include <mkl.h>
// #include <mkl_service.h>
#include "../common.h"

float kmeans(float* trainData, int numTrainData, int dim, float* codebook, int numCentroids, int* assign);
void rand_perm(int* perm, size_t n, int64_t seed);
void fillWithRandom(float* data, int size);

struct GPUIndex {
    int** indices;
    int* sizes;
    int numCoarseCentroids;
    int numFineCentroids;
};

// 检查 CUDA API 调用的宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

// 分配并拷贝索引数据到GPU端
GPUIndex copyIndexToGPU(const std::vector<std::vector<std::vector<idx_t>>>& index, int numCoarseCentroids, int numFineCentroids) {
    GPUIndex gpuIndex;
    gpuIndex.numCoarseCentroids = numCoarseCentroids;
    gpuIndex.numFineCentroids = numFineCentroids;

    // 分配指针数组
    int** hostIndices = new int*[numCoarseCentroids * numFineCentroids];
    int* hostSizes = new int[numCoarseCentroids * numFineCentroids];

    // 分配数据并拷贝到GPU
    for (int i = 0; i < numCoarseCentroids; ++i) {
        for (int j = 0; j < numFineCentroids; ++j) {
            int idx = i * numFineCentroids + j;
            hostSizes[idx] = index[i][j].size();
            if (hostSizes[idx] > 0) {
                CUDA_CHECK(cudaMalloc(&hostIndices[idx], hostSizes[idx] * sizeof(idx_t)));
                CUDA_CHECK(cudaMemcpy(hostIndices[idx], index[i][j].data(), hostSizes[idx] * sizeof(idx_t), cudaMemcpyHostToDevice));
            } else {
                hostIndices[idx] = nullptr;
            }
        }
    }

    // 分配GPU端指针
    int** deviceIndices;
    CUDA_CHECK(cudaMalloc(&deviceIndices, numCoarseCentroids * numFineCentroids * sizeof(int*)));
    int* deviceSizes;
    CUDA_CHECK(cudaMalloc(&deviceSizes, numCoarseCentroids * numFineCentroids * sizeof(int)));

    // 拷贝指针数组到GPU
    CUDA_CHECK(cudaMemcpy(deviceIndices, hostIndices, numCoarseCentroids * numFineCentroids * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSizes, hostSizes, numCoarseCentroids * numFineCentroids * sizeof(int), cudaMemcpyHostToDevice));

    // 设置 GPUIndex 的成员
    gpuIndex.indices = deviceIndices;
    gpuIndex.sizes = deviceSizes;

    // 释放临时数组
    delete[] hostIndices;
    delete[] hostSizes;

    return gpuIndex;
}

// 释放GPU端的索引数据
void freeGPUIndex(GPUIndex& gpuIndex) {
    int totalClusters = gpuIndex.numCoarseCentroids * gpuIndex.numFineCentroids;
    int* hostSizes = new int[totalClusters];
    
    cudaMemcpy(hostSizes, gpuIndex.sizes, totalClusters * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalClusters; ++i) {
        if (hostSizes[i] > 0) {
            cudaFree(gpuIndex.indices[i]);
        }
    }

    delete[] hostSizes;

    cudaFree(gpuIndex.indices);
    cudaFree(gpuIndex.sizes);
}

class RVQ {
public:
    // 构造函数
    RVQ(int dim, int numCoarseCentroids, int numFineCentroids)
        : dim_(dim), numCoarseCentroid_(numCoarseCentroids), numFineCentroid_(numFineCentroids) {
        coarseCodebook_ = new float[dim * numCoarseCentroids];
        fineCodebook_ = new float[dim * numFineCentroids];
        std::cout << "RVQ index created.\nDimensions = " << dim << ", Coarse Centroids = " << numCoarseCentroids << ", Fine Centroids = " << numFineCentroids << std::endl;
        index_.resize(numCoarseCentroids);
        for (int i = 0; i < numCoarseCentroids; ++i) {
            index_[i].resize(numFineCentroids);
            for (int j = 0; j < numFineCentroids; ++j) {
                index_[i][j].resize(0);
            }
        }

        d_index_ = nullptr;
        d_coarse_codebook_ = nullptr;
        d_fine_codebook_ = nullptr;
    }

    // 析构函数
    ~RVQ() {
        delete[] coarseCodebook_;
        delete[] fineCodebook_;
        if (d_index_) {
            freeGPUIndex(*d_index_);
            delete d_index_;
        }
        cudaFree(d_coarse_codebook_);
        cudaFree(d_fine_codebook_);
        std::cout << "RVQ object destroyed." << std::endl;
    }

    // 训练粗略量化码本
    void train(float* trainVectorData, idx_t numVectors);

    // 构建精细量化码本
    void build(float* buildVectorData, idx_t numVectors) ;

    // 查询搜索
    void search(float* query, int numQueries, int* cluster);

    void save(const std::string& filename);
    void load(const std::string& filename);

    std::vector<std::vector<std::vector<idx_t>>> get_index();
    GPUIndex* get_gpu_index();

private:
    int dim_; // 特征向量维度
    float* coarseCodebook_; // 粗码本
    float* d_coarse_codebook_; // GPU上的粗码本
    int numCoarseCentroid_; // 粗码本中心点数量
    float* fineCodebook_; // 细码本
    float* d_fine_codebook_; // GPU上的细码本
    int numFineCentroid_; // 细码本中心点数量
    std::vector<std::vector<std::vector<idx_t>>> index_;
    GPUIndex* d_index_; // GPU的索引数据
};