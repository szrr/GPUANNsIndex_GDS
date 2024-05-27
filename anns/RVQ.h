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
#include "common.h"

class RVQ {
public:
    // 构造函数
    RVQ(int dim, int numCoarseCentroids, int numFineCentroids)
        : dim_(dim), numCoarseCentroid_(numCoarseCentroids), numFineCentroid_(numFineCentroids) {
        coarseCodebook_ = new float[dim * numCoarseCentroids];
        fineCodebook_ = new float[dim * numFineCentroids];
        std::cout << "RVQ object created with dimensions: " << dim << ", Coarse Centroids: " << numCoarseCentroids << ", Fine Centroids: " << numFineCentroid << std::endl;
        index.resize(numCoarseCentroids);
        for (int i = 0; i < numCoarseCentroids; ++i) {
            index[i].resize(numFineCentroids);
            for (int j = 0; j < numFineCentroids; ++j) {
                index[i][j].resize(0);
            }
        }
    }

    // 析构函数
    ~RVQ() {
        delete[] coarseCodebook_;
        delete[] fineCodebook_;
        std::cout << "RVQ object destroyed." << std::endl;
    }

    // 训练粗略量化码本
    void train(float* trainVectorData, idx_t numVectors);

    // 构建精细量化码本
    void build(float* buildVectorData, idx_t numVectors) ;

    // 查询搜索
    void search(float* query, int numQueries, std::vector<std::pair<int, int>> res);

private:
    int dim_; // 特征向量维度
    float* coarseCodebook_; // 粗码本
    int numCoarseCentroid_; // 粗略码本中心点数量
    float* fineCodebook_; // 细码本
    int numFineCentroid_; // 精细码本中心点数量
    std::vector<std::vector<std::vector<idx_t>>> index;
};

int main() {
    RVQ rvq(128, 256, 512);
    // float trainData[] = { /* 假设为训练数据 */;
    // idx_t numTrainVectors = sizeof(trainData) / 128; // 假设数据为128维
    // rvq.train(trainData, numTrainVectors);

    // float buildData[] = { /* 建设为构建数据 */;
    // idx_t numBuildVectors = sizeof(buildData) / 128;
    // rvq.build(buildData, numBuildVectors);

    // float query[] = { /* 假设为查询数据 */;
    // rvq.search(query, sizeof(query) / 128);

    return 0;
}