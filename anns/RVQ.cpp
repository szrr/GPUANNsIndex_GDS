/**
 * @author szr
 * @date 2024/5/24
 * @brief two-layer k-means using RVQ method
 * 
 * **/

#include <memory>
#include <cstring>
#include <cmath>
#include <limits>
#include <cblas.h>
#include "RVQ.h"
#include "common.h"
#include "anns/functions/distance_kernel.cuh"
#include "anns/functions/selectMin1.cuh"

float kmeans(float* trainData, int numTrainData, int dim, float* codebook, int numCentroids, int* assign) {
    float error = 0.0;

    // 初始化聚类中心为前 numCentroids 个训练数据点
    for (int i = 0; i < numCentroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            codebook[i * dim + d] = trainData[i * dim + d];
        }
    }

    // 迭代更新聚类中心，假设收敛条件为聚类中心不再变化
    bool converged = false;
    while (!converged) {
        converged = true;

        // 对每个训练数据进行分配，并更新聚类中心
        int* counts = new int[numCentroids](); // 用于计算每个聚类的数据点数目
        float* newCentroids = new float[numCentroids * dim](); // 存储新的聚类中心的临时数组

        for (int i = 0; i < numTrainData; ++i) {
            float minDist = std::numeric_limits<float>::max();
            int minIndex = -1;

            // 计算训练数据点与每个聚类中心的距离，找到最近的聚类中心
            for (int j = 0; j < numCentroids; ++j) {
                float dist = 0.0;
                for (int d = 0; d < dim; ++d) {
                    float diff = trainData[i * dim + d] - codebook[j * dim + d];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = j;
                }
            }

            // 更新分配
            assign[i] = minIndex;

            // 更新聚类中心的临时数组
            for (int d = 0; d < dim; ++d) {
                newCentroids[minIndex * dim + d] += trainData[i * dim + d];
            }
            counts[minIndex]++;

            // 累加误差
            error += minDist;
        }

        // 计算新的聚类中心并检查是否收敛
        for (int i = 0; i < numCentroids; ++i) {
            if (counts[i] > 0) {
                for (int d = 0; d < dim; ++d) {
                    float newCentroid = newCentroids[i * dim + d] / counts[i];
                    if (codebook[i * dim + d] != newCentroid) {
                        converged = false; // 聚类中心有变化，继续迭代
                    }
                    codebook[i * dim + d] = newCentroid;
                }
            }
        }

        // 释放临时数组的内存
        delete[] counts;
        delete[] newCentroids;
    }

    return error;
}

// 训练粗略量化码本
void RVQ::train(float* trainVectorData, int numTrainVectors) {
    std::cout << "Training quantization codebook with " << numTrainVectors << " vectors." << std::endl;
    // point属于的cluster id
    // std::unique_ptr<int[]> cluster_assign(new int[numTrainVectors]);
    // 迭代的最小误差
    float min_err = std::numeric_limits<float>::max();
    // 每次k-means的训练数据 根据粗细码本迭代更新
    std::unique_ptr<float[]> trainData(new float[numTrainVectors * dim_]);
    memcpy(trainData.get(), trainVectorData, sizeof(float) * numTrainVectors * dim_);
    // 每次kmeans产生的聚类中心
    std::unique_ptr<float[]> coarseCodebook(new float[numCoarseCentroid_ * dim_]);
    std::unique_ptr<int[]> coarseCodebookAssign(new int[numTrainVectors]);
    std::unique_ptr<float[]> fineCodebook(new float[numFineCentroid_ * dim_]);
    std::unique_ptr<int[]> fineCodebookAssign(new int[numTrainVectors]);

    // 第一层 迭代训练数据 重复调用k-means 参考Tinker
    int iter = 10;
    int niter = 30;
    for (int i = 0; i < iter; ++i) {
        // 使用数据点与二级聚类中心的残差 训练一级聚类中心
        // 第一轮可以理解为二级聚类初始为空
        float err = kmeans(trainData.get(), numTrainVectors, dim_, coarseCodebook.get(), numCoarseCentroid_, coarseCodebookAssign.get());
        std::cout << iter << " deviation error of coarse clusters is " << err << " when ite = " << iter << std::endl;

        // 计算数据点与一级聚类中心的残差 训练二级聚类中心
        memcpy(trainData.get(), trainVectorData, sizeof(float) * numTrainVectors * dim_);
        for (int i = 0; i < numTrainVectors; ++i) {
            int assign_id = coarseCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, coarseCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

        err = kmeans(trainData.get(), numTrainVectors, dim_, fineCodebook.get(), numFineCentroid_, fineCodebookAssign.get());
        std::cout << iter << " deviation error of fine clusters is " << err << " when ite = " << iter << std::endl;

        // 记录最小误差的训练结果 判定训练是否收敛
        if ((min_err - err) >= 1e-4) { // 参数正在收敛
            memcpy(coarseCodebook_, coarseCodebook.get(),
                   sizeof(float) * numCoarseCentroid_ * dim_);
            memcpy(fineCodebook_, fineCodebook.get(),
                   sizeof(float) * numFineCentroid_ * dim_);
            min_err = err;
        } else { // 开始出现抖动
            std::cout << "current deviation error > min deviation error : " << err << " / " << min_err <<
                      ", params.niter = " << niter << std::endl;

            // niter初值为30，大于80跳出
            if (niter > 80) {
                break;
            }

            niter += 10;
        }

        // 计算数据点与二级聚类中心的残差
        memcpy(trainData.get(), trainVectorData, sizeof(float) * numTrainVectors * dim_);
        for (int i = 0; i < numTrainVectors; ++i) {
            //每次迭代计算T之后的值作为判断标准，所以每次S的值使用最新计算的
            int assign_id = fineCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, fineCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

    }

    // 第二层 k-means训练 随机生成聚类中心多轮训练 参考Tinker
    // 第三层 k-means聚类中心计算和迭代 参考Faiss IVF


    
}

// 构建反向索引
void RVQ::build(float* buildVectorData, num_t numVectors) {
    std::cout << "Building fine quantization codebook with " << numVectors << " vectors." << std::endl;
    // Todo: 距离计算没有分块，可能放不下
    // Todo: fuse (distance + 1-selection) 放入一个kernel?

    // 计算与粗聚类中心距离
    float* disMatrixCoarse = new float[numVectors * numCoarseCentroid_];
    QueryToBaseDistance(coarseCodebook_, numCoarseCentroid_, buildVectorData,
                         numVectors, dim_, disMatrixCoarse, 100000);
    
    // 得到最近的粗聚类中心
    std::vector<int> minCoarseIndices = findMinIndicesCUDA(disMatrixCoarse, numCoarseCentroid_, numVectors);
    
    // 计算残差
    std::unique_ptr<float[]> FineData(new float[numVectors * dim_]);
    memcpy(FineData.get(), buildVectorData, sizeof(float) * numVectors * dim_);
    for (int i = 0; i < numVectors; ++i) {
        int assign_id = minCoarseIndices.data()[i];
        cblas_saxpy(dim_, -1.0, fineCodebook_ + assign_id * dim_, 1,
                    FineData.get() + i * dim_, 1);
    }

    // 得到最近的细聚类中心
    float* disMatrixFine = new float[numVectors * numFineCentroid_];
    QueryToBaseDistance(fineCodebook_, numFineCentroid_, buildVectorData,
                         numVectors, dim_, disMatrixFine, 100000);
    std::vector<int> minFineIndices = findMinIndicesCUDA(disMatrixFine, numFineCentroid_, numVectors);

    // 加入反向列表
    for(idx_t i = 0; i < numVectors; ++i){
        index[minCoarseIndices[i]][minFineIndices[i]].push_back(i);
    }
}

// 查询搜索
void RVQ::search(float* query, int numQueries, std::vector<std::pair<int, int>> res) {
    std::cout << "Searching with " << numQueries << " queries." << std::endl;
    // Todo: search()
    // 找到两层最近邻 过程参考Tinker 计算参考Faiss
    
    // 计算与粗聚类中心距离
    float* disMatrixCoarse = new float[numQueries * numCoarseCentroid_];
    QueryToBaseDistance(coarseCodebook_, numCoarseCentroid_, query,
                         numQueries, dim_, disMatrixCoarse, 100000);
    
    // 得到最近的粗聚类中心
    std::vector<int> minCoarseIndices = findMinIndicesCUDA(disMatrixCoarse, numCoarseCentroid_, numQueries);
    
    // 计算残差
    std::unique_ptr<float[]> FineData(new float[numQueries * dim_]);
    memcpy(FineData.get(), query, sizeof(float) * numQueries * dim_);
    for (int i = 0; i < numQueries; ++i) {
        int assign_id = minCoarseIndices.data()[i];
        cblas_saxpy(dim_, -1.0, fineCodebook_ + assign_id * dim_, 1,
                    FineData.get() + i * dim_, 1);
    }

    // 得到最近的细聚类中心
    float* disMatrixFine = new float[numQueries * numFineCentroid_];
    QueryToBaseDistance(fineCodebook_, numFineCentroid_, query,
                         numQueries, dim_, disMatrixFine, 100000);
    std::vector<int> minFineIndices = findMinIndicesCUDA(disMatrixFine, numFineCentroid_, numQueries);

    //Todo: 得到minCoarseIndices和minFineIndices之后，需要进行什么操作？返回index？
    for(idx_t i = 0; i < numQueries; ++i){
        res.push_back(std::make_pair(minCoarseIndices[i], minFineIndices[i]));
    }
}

