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
#include <random>
#include <iostream>
#include <vector>
#include <random>
// #include </usr/include/mkl/mkl_cblas.h>
// #include </usr/include/mkl/mkl.h>
// #include </usr/include/mkl/mkl_service.h>
#include "RVQ.h"
#include "../common.h"
#include "../functions/distance_kernel.cuh"
#include "../functions/selectMin1.cuh"

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

// rand_perm 函数实现
void rand_perm(int* perm, size_t n, int64_t seed) {
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    std::mt19937 rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        size_t i2 = dist(rng);
        std::swap(perm[i], perm[i2]);
    }
}

// subsample_training_set 函数实现
void subsample_training_set(
    const float* dataset, 
    idx_t numVector, 
    int dim, 
    idx_t numTrainVec, 
    float* trainVectors,
    int64_t seed = 1234) {

    // 生成随机排列
    std::vector<int> perm(numVector);
    rand_perm(perm.data(), numVector, seed);
    

    // 选择前 numTrainVec 个随机向量作为训练数据
    for (idx_t i = 0; i < numTrainVec; i++) {
        std::memcpy(trainVectors + i * dim, dataset + perm[i] * dim, sizeof(float) * dim);
    }
}

std::vector<int> findMinIndicesCPU(const float* values, int c, int num) {
    std::vector<int> minIndices(num);
    for (int idx = 0; idx < num; ++idx) {
        int startIdx = idx * c;
        int endIdx = (idx + 1) * c;
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
    return minIndices;
}

float Kmeans (float* trainData, idx_t numTrainData, int dim, float* codebook, int numCentroids, int* assign) {
    printf("[Info] Start Kmeans training\n");

    float* bestCentroids = new float[numCentroids * dim];

    int iter = 25;
    float* centroids = new float[numCentroids * dim];
    memcpy(centroids, trainData, numCentroids * dim * sizeof(float));
    float* disMatrix = new float[numCentroids * numTrainData];
    std::vector<int> minCentroids;
    std::vector<int> bestMinCentroids(numTrainData, 0);
    std::vector<int> countCentroidPoints(numCentroids, 0);

    float err;
    float minErr = std::numeric_limits<float>::max();

    // loop
    for (int i = 0; i < iter; ++i) {
        printf("Training iteration %d. ", i);
        // distance calculation
        queryToBaseDistance(centroids, numCentroids, trainData, numTrainData, dim, disMatrix);

        // find the nearest centroids of data points
        minCentroids = findMinIndices(disMatrix,  numCentroids, numTrainData);
        // // 验证
        // std::vector<int> cpuResults = findMinIndicesCPU(disMatrix, numCentroids, numTrainData);
        // for (int i = 0; i < numTrainData; ++i) {
        //     if (cpuResults[i] != minCentroids[i]) {
        //         std::cerr << "Mismatch at index " << i << ": CPU result = " << cpuResults[i] << ", GPU result = " << minCentroids[i] << std::endl;
        //     }
        // }

        std::fill(countCentroidPoints.begin(), countCentroidPoints.end(), 0);
        for (int i = 0; i < numTrainData; ++i) {
            countCentroidPoints[minCentroids[i]]++;
        }

        // printf("Points of each centroid: \n");
        // for (int i = 0; i < numCentroids; ++i) {
        //     printf("centroid%d : %d ", i, countCentroidPoints[i]);
        // }
        // printf("\n");

        // printf("Total %d centroids, minCentroids\n", numCentroids);
        // for (int i = 0; i < 20; ++i) {
        //     printf("[%d] = %d ", i, minCentroids[i]);
        // }
        // printf("\n");

        // record the best centroids with min error
        err = 0.0;
        for (int i = 0; i < numTrainData; ++i) {
            err += disMatrix[i*numCentroids + minCentroids[i]];
        }
        // printf("error = %.2f\n", err);
        if (err < minErr) {
            // error < minErr, update minErr, record bestCentroids and bestMinCentroids
            printf("error = %.2f < minErr = %.2f, update minErr\n", err, minErr);
            minErr = err;
            memcpy(bestCentroids, centroids, numCentroids * dim * sizeof(float));
            memcpy(bestMinCentroids.data(), minCentroids.data(), numTrainData * sizeof(int));
        }

        // 用聚类中数据点的平均值更新聚类中心
        // 如果聚类中心里没有数据点，则保留；如果有，则使用数据点平均值代替它
        // printf("[Info] Centroids with no cluster point in it:");
        for (int i = 0; i < numCentroids; ++i) {
            // printf("[%d] = %d ", i, countCentroidPoints[i]);
            if (countCentroidPoints[i] == 0) {
                // printf("centroid[%d] ", i);
            } else {
                memset(centroids + i*dim, 0, dim*sizeof(float));
                // centroids[i] = 0;
            }
        }
        // printf("\n");

        for (int i = 0; i < numTrainData; ++i) {
            for (int j = 0; j < dim; ++j) {
                centroids[minCentroids[i]*dim+j] += (trainData[i*dim+j] / countCentroidPoints[minCentroids[i]]);
            }
        }
    }
    printf("[Info] Finish Kmeans training\n");
    memcpy(codebook, bestCentroids, numCentroids * dim * sizeof(float));
    memcpy(assign, minCentroids.data(), numTrainData * sizeof(int));
    
    return minErr;
}

// 训练粗略量化码本
void RVQ::train(float* trainVectorData, idx_t numTrainVectors) {
    std::cout << "Training input : " << numTrainVectors << " vectors." << std::endl;

    // 采样训练点，不超过10w
    idx_t numSelectTrainVec = 100000;
    // 检查输入参数是否有效
    if (numSelectTrainVec > numTrainVectors) {
        std::cout << "Number of select training vectors : " << numTrainVectors << std::endl;
        numSelectTrainVec = numTrainVectors;
    } else {
        std::cout << "Number of select training vectors : 100000" << std::endl;
    }

    float* selectTrainVectors = new float[numSelectTrainVec * dim_];
    subsample_training_set(trainVectorData, numTrainVectors, dim_, numSelectTrainVec, selectTrainVectors);

    // 迭代的最小误差
    float min_err = std::numeric_limits<float>::max();
    // 每次k-means的训练数据 根据粗细码本迭代更新
    std::unique_ptr<float[]> trainData(new float[numSelectTrainVec * dim_]);
    memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
    // 每次kmeans产生的聚类中心
    std::unique_ptr<float[]> coarseCodebook(new float[numCoarseCentroid_ * dim_]);
    std::unique_ptr<int[]> coarseCodebookAssign(new int[numTrainVectors]);
    std::unique_ptr<float[]> fineCodebook(new float[numFineCentroid_ * dim_]);
    std::unique_ptr<int[]> fineCodebookAssign(new int[numTrainVectors]);

    // 第一层 迭代训练数据 重复调用k-means 参考Tinker
    int iter = 10;
    int niter = 30;
    for (int i = 0; i < iter; ++i) {
        printf("[Info] RVQ training iteration %d\n", i);
        // 使用数据点与二级聚类中心的残差 训练一级聚类中心
        // 第一轮可以理解为二级聚类初始为空
        // Todo: debug Kmeans(), the coarse centroids become the same value after some iterations
        printf("[Info] Training coarse codebook\n");
        float err = Kmeans(trainData.get(), numSelectTrainVec, dim_, coarseCodebook.get(), numCoarseCentroid_, coarseCodebookAssign.get());
        std::cout << "The" << i << " deviation error of coarse clusters is " << err << std::endl;
        std::cout << std::endl;

        // 计算数据点与一级聚类中心的残差 训练二级聚类中心
        memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
        for (int i = 0; i < numSelectTrainVec; ++i) {
            int assign_id = coarseCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, coarseCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

        printf("[Info] Training fine codebook\n");
        err = Kmeans(trainData.get(), numSelectTrainVec, dim_, fineCodebook.get(), numFineCentroid_, fineCodebookAssign.get());
        std::cout << "The " << i << " deviation error of fine clusters is " << err << std::endl;
        std::cout << std::endl;

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
        memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
        for (idx_t i = 0; i < numSelectTrainVec; ++i) {
            //每次迭代计算T之后的值作为判断标准，所以每次S的值使用最新计算的
            int assign_id = fineCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, fineCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

    }
    
}

// 构建反向索引
void RVQ::build(float* buildVectorData, num_t numVectors) {
    std::cout << "Building index with " << numVectors << " build vectors." << std::endl;
    // Todo: 距离计算没有分块，可能放不下
    // Todo: fuse (distance + 1-selection) 放入一个kernel?

    // 计算与粗聚类中心距离
    float* disMatrixCoarse = new float[numVectors * numCoarseCentroid_];
    queryToBaseDistance(coarseCodebook_, numCoarseCentroid_, buildVectorData,
                         numVectors, dim_, disMatrixCoarse, 100000);
    
    // 得到最近的粗聚类中心
    std::vector<int> minCoarseIndices = findMinIndices(disMatrixCoarse, numCoarseCentroid_, numVectors);
    
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
    queryToBaseDistance(fineCodebook_, numFineCentroid_, buildVectorData,
                         numVectors, dim_, disMatrixFine, 100000);
    std::vector<int> minFineIndices = findMinIndices(disMatrixFine, numFineCentroid_, numVectors);

    // 加入反向列表
    for(idx_t i = 0; i < numVectors; ++i){
        index[minCoarseIndices[i]][minFineIndices[i]].push_back(i);
        if (i < 100) {
            printf("dataset%d : coarseId=%d, fineId=%d\n", i, minCoarseIndices[i], minFineIndices[i]);
        }
    }
    printf("\n");
}

// 查询搜索
void RVQ::search(float* query, int numQueries, std::vector<std::vector<idx_t>>& res) {
    std::cout << "Searching with " << numQueries << " queries." << std::endl;

    // 计算与粗聚类中心距离
    float* disMatrixCoarse = new float[numQueries * numCoarseCentroid_];
    queryToBaseDistance(coarseCodebook_, numCoarseCentroid_, query,
                         numQueries, dim_, disMatrixCoarse, 100000);
    
    // 得到最近的粗聚类中心
    std::vector<int> minCoarseIndices = findMinIndices(disMatrixCoarse, numCoarseCentroid_, numQueries);
    
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
    queryToBaseDistance(fineCodebook_, numFineCentroid_, query,
                         numQueries, dim_, disMatrixFine, 100000);
    std::vector<int> minFineIndices = findMinIndices(disMatrixFine, numFineCentroid_, numQueries);

    //Todo: 得到minCoarseIndices和minFineIndices之后，需要进行什么操作？返回index？
    for(idx_t i = 0; i < numQueries; ++i){
        res.push_back(index[minCoarseIndices[i]][minFineIndices[i]]);
        // if (i < 100){
        //     printf("Query %d : coarseId = %d, fineId = %d\n", i, minCoarseIndices[i], minFineIndices[i]);
        // }
    }
}

// Function to fill an array with random values
void fillWithRandom(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen)*100;
    }
}

// int main() {
//     // Define parameters
//     int dim = 128; // Dimension of feature vectors
//     int numCoarseCentroids = 100; // Number of coarse centroids
//     int numFineCentroids = 100; // Number of fine centroids
//     int numTrainVectors = 100000; // Number of training vectors
//     int numQueries = 10000; // Number of query vectors

//     // Generate random training data and query data
//     float* trainData = new float[numTrainVectors * dim];
//     float* queryData = new float[numQueries * dim];
//     fillWithRandom(trainData, numTrainVectors * dim);
//     fillWithRandom(queryData, numQueries * dim);

//     // Create RVQ object
//     RVQ rvq(dim, numCoarseCentroids, numFineCentroids);

//     // Train RVQ
//     rvq.train(trainData, numTrainVectors);

//     for(int i = 0; i < 10; ++i){
//         printf("coarse centroid [%d]:", i);
//         for(int j = 0; j < dim; j++){
//             printf("%f ", rvq.coarseCodebook_[i*dim + j]);
//         }
//         printf("\nfine centroid [%d]:", i);
//         for(int j = 0; j < dim; j++){
//             printf("%f ", rvq.fineCodebook_[i*dim + j]);
//         }
//         printf("\n");
//     }

//     // Build reverse index
//     rvq.build(trainData, numTrainVectors);

//     // Search using queries
//     std::vector<std::pair<int, int>> results;
//     rvq.search(queryData, numQueries, results);

//     // Display search results
//     std::cout << "Search results:" << std::endl;
//     for (int i = 0; i < 100; ++i) {
//         std::cout << "Query " << i << ": Coarse index = " << results[i].first << ", Fine index = " << results[i].second << std::endl;
//     }

//     // Clean up
//     delete[] trainData;
//     delete[] queryData;

//     return 0;
// }