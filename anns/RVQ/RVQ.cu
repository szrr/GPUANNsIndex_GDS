/**
 * @author szr
 * @date 2024/5/24
 * @brief two-layer k-means using RVQ method
 * 
 * **/

#include <memory>
#include <fstream>
#include <cstring>
#include <cmath>
#include <limits>
#include <cblas.h>
#include <random>
#include <iostream>
#include <vector>
#include <random>
#include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include </usr/include/mkl/mkl_cblas.h>
// #include </usr/include/mkl/mkl.h>
// #include </usr/include/mkl/mkl_service.h>
#include "RVQ.cuh"
#include "../common.h"
#include "../functions/check.h"
#include "../functions/distance_kernel.cuh"
#include "../functions/selectMin1.cuh"


void checkPointerType(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    switch (attributes.type) {
        case cudaMemoryTypeHost:
            std::cout << "Pointer is a host (CPU) pointer." << std::endl;
            break;
        case cudaMemoryTypeDevice:
            std::cout << "Pointer is a device (GPU) pointer." << std::endl;
            break;
        case cudaMemoryTypeManaged:
            std::cout << "Pointer is a managed memory pointer." << std::endl;
            break;
        default:
            std::cout << "Pointer type is unknown." << std::endl;
            break;
    }
}

// 分配并拷贝索引数据到GPU端
void copyIndexToGPU(const std::vector<std::vector<std::vector<idx_t>>>& index, int numCoarseCentroids, int numFineCentroids, GPUIndex* d_index) {

    d_index->numCoarseCentroids = numCoarseCentroids;
    d_index->numFineCentroids = numFineCentroids;

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
                CUDA_CHECK(cudaMemcpy(hostIndices[idx], &index[i][j][0], hostSizes[idx] * sizeof(idx_t), cudaMemcpyHostToDevice));
            } else {
                hostIndices[idx] = nullptr;
            }
        }
    }

     // 分配GPU端指针
    CUDA_CHECK(cudaMalloc(&d_index->indices, numCoarseCentroids * numFineCentroids * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&d_index->sizes, numCoarseCentroids * numFineCentroids * sizeof(int)));

    // 拷贝指针数组到GPU
    CUDA_CHECK(cudaMemcpy(d_index->indices, hostIndices, numCoarseCentroids * numFineCentroids * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_index->sizes, hostSizes, numCoarseCentroids * numFineCentroids * sizeof(int), cudaMemcpyHostToDevice));

    // 释放临时数组
    delete[] hostIndices;
    delete[] hostSizes;
}

// 释放GPU端的索引数据
void freeGPUIndex(GPUIndex& gpuIndex) {
    if (gpuIndex.indices != nullptr) {
        int totalClusters = gpuIndex.numCoarseCentroids * gpuIndex.numFineCentroids;
        int* hostSizes = new int[totalClusters];
        
        int** hostIndices = new int*[totalClusters];
        CUDA_CHECK(cudaMemcpy(hostIndices, gpuIndex.indices, totalClusters * sizeof(int*), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(hostSizes, gpuIndex.sizes, totalClusters * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < totalClusters; ++i) {
            if (hostSizes[i] > 0) {
                CUDA_CHECK(cudaFree(hostIndices[i]));
            }
        }

        delete[] hostSizes;

        checkPointerType(gpuIndex.indices);
        checkPointerType(gpuIndex.sizes);
        CUDA_CHECK(cudaFree(gpuIndex.indices));
        CUDA_CHECK(cudaFree(gpuIndex.sizes));
    }
}

// GPU 计算最终clusterId的kernel
__global__ void addKernel(int* d_min_coarse_indices, int* d_min_fine_indices, int* d_enter_cluster, int numFineCentroid, int numQueries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numQueries) {
        d_enter_cluster[idx] = d_min_coarse_indices[idx] * numFineCentroid + d_min_fine_indices[idx];
    }
}

// // GPU 计算残差的 kernel
// __global__ void computeResiduals(const float* coarse_codebook, const int* min_coarse_indices, float* fine_data, int dim, int num_queries, cublasHandle_t handle) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num_queries) {
//         int assign_id = min_coarse_indices[idx];
//         const float alpha = -1.0f;
//         cublasSaxpy(handle, dim, &alpha, coarse_codebook + assign_id * dim, 1, fine_data + idx * dim, 1);
//     }
// }

// 手动计算残差的 kernel
__global__ void computeResiduals(const float* coarse_codebook, const int* min_coarse_indices, float* fine_data, int dim, int num_queries) {
    int query_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (query_idx < num_queries && dim_idx < dim) {
        int assign_id = min_coarse_indices[query_idx];
        fine_data[query_idx * dim + dim_idx] -= coarse_codebook[assign_id * dim + dim_idx];
    }
}


std::mt19937 _rnd(time(0));

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

// Function to fill an array with random values
void fillWithRandom(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen)*100;
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

int rouletteSelection(std::vector<float>& wheel) {
    float total_val = 0;

    for (auto& val : wheel) {
        total_val += val;
    }

    cblas_sscal(wheel.size(), 1.0 / total_val, wheel.data(), 1);
    std::uniform_real_distribution<double> dis(0, 1.0);
    double rd = dis(_rnd);

    for (auto id = 0; id < wheel.size();  ++id) {
        rd -= wheel[id];

        if (rd < 0) {
            return id;
        }
    }

    return wheel.size() - 1;
}

void KmeansppInitCenter(const int total_cnt, const int sample_cnt,
                                  const int dim,
                                  const float* train_dataset, std::vector<int>& sample_ids) {
    std::vector<float> disbest(total_cnt, std::numeric_limits<float>::max()); // 每个点到最近中心的最小距离
    std::vector<float> distmp(total_cnt); // 临时距离存储
    sample_ids.resize(sample_cnt, 0); // 初始化中心索引数组
    sample_ids[0] = _rnd() % total_cnt; // 随机选择第一个中心

    std::vector<float> points_norm(total_cnt, 0); // 每个点的范数

    // 计算每个数据点的范数
    #pragma omp parallel for schedule(dynamic) num_threads(_params.nt)
    for (size_t j = 0; j < total_cnt; j++) {
        points_norm[j] = cblas_sdot(dim, train_dataset + j * dim, 1, train_dataset + j * dim, 1);
    }

    // 选择剩余的初始中心
    for (size_t i = 1; i < sample_cnt; i++) {
        size_t newsel = sample_ids[i - 1]; // 当前选择的中心索引
        const float* last_center = train_dataset + newsel * dim; // 当前选择的中心数据

        #pragma omp parallel for schedule(dynamic) num_threads(_params.nt)
        for (size_t j = 0; j < total_cnt; j++) {
            float temp = points_norm[j] + points_norm[newsel] - 2.0 * cblas_sdot(dim, train_dataset + j * dim, 1,
                         last_center, 1); // 计算距离平方

            if (temp < disbest[j]) {
                disbest[j] = temp; // 更新最近中心的距离
            }
        }

        memcpy(distmp.data(), disbest.data(), total_cnt * sizeof(distmp[0])); // 复制距离数据
        sample_ids[i] = rouletteSelection(distmp); // 轮盘赌选择下一个中心
    }
}

float Kmeans (float* trainData, idx_t numTrainData, int dim, float* codebook, int numCentroids, int* assign) {
    printf("[Info] Start Kmeans training\n");

    // 使用 K-means++ 初始化中心
    std::vector<int> sample_ids;
    KmeansppInitCenter(numTrainData, numCentroids, dim, trainData, sample_ids);

    // 初始化聚类中心为 K-means++ 选择的中心
    for (int i = 0; i < numCentroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            codebook[i * dim + d] = trainData[sample_ids[i] * dim + d];
        }
    }

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

std::vector<std::vector<std::vector<idx_t>>> RVQ::get_index(){
    return index_;
}

GPUIndex* RVQ::get_gpu_index() {
    return d_index_;
}

// 训练粗略量化码本
void RVQ::train(float* trainVectorData, idx_t numTrainVectors) {
    std::cout << "Training input : " << numTrainVectors << " vectors." << std::endl;
    // for(int i = 0; i<numTrainVectors; i++){
    //     std::cout<<i<<std::endl;
    //     for(int l=0; l<128; l++){
    //         std::cout<<trainVectorData[i*128 + l]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
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

__global__ void testKernel(int** d_indices, int* d_sizes, int* output, int* sizes_output, int numCoarseCentroids, int numFineCentroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCoarseCentroids * numFineCentroids) {
        int size = d_sizes[idx];
        sizes_output[idx] = size;
        if (size > 0) {
            output[idx] = d_indices[idx][0]; // 只读取每个cluster中的第一个元素进行测试
        } else {
            output[idx] = -1; // 标记为空的情况
        }
    }
}
void testIndices(GPUIndex* d_index, int numCoarseCentroids, int numFineCentroids) {
    int totalClusters = numCoarseCentroids * numFineCentroids;
    int* h_output = new int[totalClusters];
    int* h_sizes_output = new int[totalClusters];
    int* d_output;
    int* d_sizes_output;
    cudaMalloc(&d_output, totalClusters * sizeof(int));
    cudaMalloc(&d_sizes_output, totalClusters * sizeof(int));

    // 启动核函数
    int blockSize = 256;
    int numBlocks = (totalClusters + blockSize - 1) / blockSize;
    testKernel<<<numBlocks, blockSize>>>(d_index->indices, d_index->sizes, d_output, d_sizes_output, numCoarseCentroids, numFineCentroids);
    cudaDeviceSynchronize();

    // 将结果从设备端拷贝到主机端
    cudaMemcpy(h_output, d_output, totalClusters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sizes_output, d_sizes_output, totalClusters * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < totalClusters; ++i) {
        std::cout << "Cluster " << i << " size: " << h_sizes_output[i];
        if (h_output[i] != -1) {
            std::cout << ", first element: " << h_output[i] << std::endl;
        } else {
            std::cout << ", is empty." << std::endl;
        }
    }

    // 释放内存
    delete[] h_output;
    delete[] h_sizes_output;
    cudaFree(d_output);
    cudaFree(d_sizes_output);
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
        cblas_saxpy(dim_, -1.0, coarseCodebook_ + assign_id * dim_, 1,
                    FineData.get() + i * dim_, 1);
    }

    // 得到最近的细聚类中心
    float* disMatrixFine = new float[numVectors * numFineCentroid_];
    queryToBaseDistance(fineCodebook_, numFineCentroid_, FineData.get(),
                         numVectors, dim_, disMatrixFine, 100000);
    std::vector<int> minFineIndices = findMinIndices(disMatrixFine, numFineCentroid_, numVectors);

    // 加入反向列表
    for(idx_t i = 0; i < numVectors; ++i){
        index_[minCoarseIndices[i]][minFineIndices[i]].push_back(i);
        if (i < 100) {
            printf("dataset%d : coarseId=%d, fineId=%d\n", i, minCoarseIndices[i], minFineIndices[i]);
        }
    }
    printf("\n");

    // 拷贝索引数据到GPU
    // freeGPUIndex(*d_index_);

    copyIndexToGPU(index_, numCoarseCentroid_, numFineCentroid_, d_index_);
    //testIndices(d_index_, numCoarseCentroid_, numFineCentroid_);

    // 拷贝粗码本和细码本到GPU
    cudaMalloc(&d_coarse_codebook_, numCoarseCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_coarse_codebook_, coarseCodebook_, numCoarseCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fine_codebook_, numFineCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_fine_codebook_, fineCodebook_, numFineCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

// 查询搜索
void RVQ::search(float* d_query, int numQueries, int* d_enter_cluster) {
    std::cout << "Searching with " << numQueries << " queries." << std::endl;

    Timer queryT;
    queryT.Start();
    // 计算与粗聚类中心距离
    //Todo: modify queryToBaseDistance() input to GPU query input
    //Todo: use GPU memory disMatrix
    float* d_dis_matrix_coarse;
    cudaMalloc((void**)&d_dis_matrix_coarse, numQueries * numCoarseCentroid_ * sizeof(float));
    deviceQueryToBaseDistance(d_coarse_codebook_, numCoarseCentroid_, d_query, numQueries, dim_, d_dis_matrix_coarse, 100000);
    queryT.Stop();
    std::cout<<"[RVQ] distance to coarse centroids time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;

    queryT.Start();
    int* d_min_coarse_indices;
    cudaMalloc((void**)&d_min_coarse_indices, numQueries * sizeof(int));
    // 得到最近的粗聚类中心
    deviceFindMinIndices(d_dis_matrix_coarse, numCoarseCentroid_, numQueries, d_min_coarse_indices);
    queryT.Stop();
    std::cout<<"[RVQ] coarse centroids findMinIndices time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    
    // 分配残差计算所需的内存
    float* d_fine_data;
    cudaMalloc((void**)&d_fine_data, numQueries * dim_ * sizeof(float));
    cudaMemcpy(d_fine_data, d_query, numQueries * dim_ * sizeof(float), cudaMemcpyDeviceToDevice);

    // 计算残差
    // Todo: change cblas_saxpy to cublasSaxpy
    queryT.Start();
    
    // // 定义 kernel 线程配置
    // int block_size = 256;
    // int num_blocks = (numQueries + block_size - 1) / block_size;

    // cublasHandle_t cublas_handle;
    // cublasCreate(&cublas_handle);
    // // 启动 kernel 计算残差
    // computeResiduals<<<num_blocks, block_size>>>(d_coarse_codebook_, d_min_coarse_indices, d_fine_data, dim_, numQueries, cublas_handle);
    // cublasDestroy(cublas_handle);

    dim3 block_size(dim_); // 每个block的线程数等于dim
    dim3 num_blocks(numQueries); // block的数量等于查询数量

    // 启动 kernel 计算残差
    computeResiduals<<<num_blocks, block_size>>>(d_coarse_codebook_, d_min_coarse_indices, d_fine_data, dim_, numQueries);
    CUDA_SYNC_CHECK();

    queryT.Stop();
    std::cout<<"[RVQ] calculate residual time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    

    queryT.Start();
    // 得到最近的细聚类中心
    float* d_dis_matrix_fine;
    cudaMalloc((void**)&d_dis_matrix_fine, numQueries * numFineCentroid_ * sizeof(float));

    deviceQueryToBaseDistance(d_fine_codebook_, numFineCentroid_, d_fine_data,
                         numQueries, dim_, d_dis_matrix_fine, 100000);
    queryT.Stop();
    std::cout<<"[RVQ] distance to fine centroids time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    
    // 得到最近的细聚类中心
    queryT.Start();
    int* d_min_fine_indices;
    cudaMalloc((void**)&d_min_fine_indices, numQueries * sizeof(int));
    deviceFindMinIndices(d_dis_matrix_fine, numFineCentroid_, numQueries, d_min_fine_indices);
    queryT.Stop();
    std::cout<<"[RVQ] fine centroids findMinIndices time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;


    //Todo: 得到minCoarseIndices和minFineIndices之后，需要进行什么操作？返回index？
    queryT.Start();
    // addKernel进行加和
    int blockSize = 256;
    int numBlocks = (numQueries + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_min_coarse_indices, d_min_fine_indices, d_enter_cluster, numFineCentroid_, numQueries);
    CUDA_SYNC_CHECK();

    // for(idx_t i = 0; i < numQueries; ++i){
    //     enter_cluster.push_back(minCoarseIndices[i] * numFineCentroid_ + minFineIndices[i]);
    //     // if (i < 100){
    //     //     printf("Query %d : coarseId = %d, fineId = %d\n", i, minCoarseIndices[i], minFineIndices[i]);
    //     // }
    // }
    queryT.Stop();
    std::cout<<"[RVQ] init result vector time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
}

void RVQ::save(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving." << std::endl;
        return;
    }

    // 保存维度、聚类中心数量
    out.write(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    out.write(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 保存粗聚类中心
    out.write(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 保存细聚类中心
    out.write(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));

    // 保存索引
    size_t outerSize = index_.size();
    out.write(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    for (const auto& inner : index_) {
        size_t innerSize = inner.size();
        out.write(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        for (const auto& innerInner : inner) {
            size_t innerInnerSize = innerInner.size();
            out.write(reinterpret_cast<char*>(&innerInnerSize), sizeof(innerInnerSize));
            out.write(reinterpret_cast<char*>(const_cast<idx_t*>(innerInner.data())), innerInnerSize * sizeof(idx_t));
        }
    }

    // 保存d_index_
    if (d_index_) {
        out.write(reinterpret_cast<char*>(&d_index_->numCoarseCentroids), sizeof(d_index_->numCoarseCentroids));
        out.write(reinterpret_cast<char*>(&d_index_->numFineCentroids), sizeof(d_index_->numFineCentroids));
        for (int i = 0; i < d_index_->numCoarseCentroids; ++i) {
            for (int j = 0; j < d_index_->numFineCentroids; ++j) {
                int idx = i * d_index_->numFineCentroids + j;
                int size = d_index_->sizes[idx];
                out.write(reinterpret_cast<char*>(&size), sizeof(size));
                if (size > 0) {
                    std::vector<idx_t> temp(size);
                    cudaMemcpy(temp.data(), d_index_->indices[idx], size * sizeof(idx_t), cudaMemcpyDeviceToHost);
                    out.write(reinterpret_cast<char*>(temp.data()), size * sizeof(idx_t));
                }
            }
        }
    } else {
        int zero = 0;
        out.write(reinterpret_cast<char*>(&zero), sizeof(zero));
        out.write(reinterpret_cast<char*>(&zero), sizeof(zero));
    }

    out.close();
}

void RVQ::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }

    // 加载维度、聚类中心数量
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    in.read(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 分配内存
    delete[] coarseCodebook_;
    delete[] fineCodebook_;
    coarseCodebook_ = new float[numCoarseCentroid_ * dim_];
    fineCodebook_ = new float[numFineCentroid_ * dim_];

    // 加载粗聚类中心
    in.read(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 加载细聚类中心
    in.read(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));

    // 加载索引
    size_t outerSize;
    in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    index_.resize(outerSize);
    for (auto& inner : index_) {
        size_t innerSize;
        in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        inner.resize(innerSize);
        for (auto& innerInner : inner) {
            size_t innerInnerSize;
            in.read(reinterpret_cast<char*>(&innerInnerSize), sizeof(innerInnerSize));
            innerInner.resize(innerInnerSize);
            in.read(reinterpret_cast<char*>(innerInner.data()), innerInnerSize * sizeof(idx_t));
        }
    }

    // 加载d_index_
    int numCoarseCentroids, numFineCentroids;
    in.read(reinterpret_cast<char*>(&numCoarseCentroids), sizeof(numCoarseCentroids));
    in.read(reinterpret_cast<char*>(&numFineCentroids), sizeof(numFineCentroids));
    if (numCoarseCentroids > 0 && numFineCentroids > 0) {
        d_index_ = new GPUIndex();
        d_index_->numCoarseCentroids = numCoarseCentroids;
        d_index_->numFineCentroids = numFineCentroids;
        CUDA_CHECK(cudaMalloc(&d_index_->indices, numCoarseCentroids * numFineCentroids * sizeof(int*)));
        CUDA_CHECK(cudaMalloc(&d_index_->sizes, numCoarseCentroids * numFineCentroids * sizeof(int)));

        std::vector<int*> hostIndices(numCoarseCentroids * numFineCentroids, nullptr);
        std::vector<int> hostSizes(numCoarseCentroids * numFineCentroids, 0);

        for (int i = 0; i < numCoarseCentroids; ++i) {
            for (int j = 0; j < numFineCentroids; ++j) {
                int idx = i * numFineCentroids + j;
                int size;
                in.read(reinterpret_cast<char*>(&size), sizeof(size));
                hostSizes[idx] = size;
                if (size > 0) {
                    CUDA_CHECK(cudaMalloc(&hostIndices[idx], size * sizeof(idx_t)));
                    std::vector<idx_t> temp(size);
                    in.read(reinterpret_cast<char*>(temp.data()), size * sizeof(idx_t));
                    CUDA_CHECK(cudaMemcpy(hostIndices[idx], temp.data(), size * sizeof(idx_t), cudaMemcpyHostToDevice));
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_index_->indices, hostIndices.data(), numCoarseCentroids * numFineCentroids * sizeof(int*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_index_->sizes, hostSizes.data(), numCoarseCentroids * numFineCentroids * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        d_index_ = nullptr;
    }

    in.close();
}

// int main() {
//     // Define parameters
//     int dim = 128; // Dimension of feature vectors
//     int numCoarseCentroids = 10; // Number of coarse centroids
//     int numFineCentroids = 10; // Number of fine centroids
//     int numTrainVectors = 1000; // Number of training vectors
//     int numQueries = 100; // Number of query vectors

//     // Generate random training data and query data
//     float* trainData = new float[numTrainVectors * dim];
//     float* queryData = new float[numQueries * dim];
//     fillWithRandom(trainData, numTrainVectors * dim);
//     fillWithRandom(queryData, numQueries * dim);

//     float *d_queries;
//     cudaMalloc((void **)&d_queries, sizeof(float) * numQueries * dim);
//     cudaMemcpy(d_queries, queryData, sizeof(float) * numQueries * dim, cudaMemcpyHostToDevice);

//     // Create RVQ object
//     RVQ rvq(dim, numCoarseCentroids, numFineCentroids);

//     // Train RVQ
//     rvq.train(trainData, numTrainVectors);

//     // Build reverse index
//     rvq.build(trainData, numTrainVectors);

//     // // Search using queries
//     int* results;
//     cudaMalloc((void**)&results, numQueries * sizeof(int));
//     rvq.search(d_queries, numQueries, results);
//     cudaDeviceSynchronize();

//     // // Copy results from device to host
//     int* h_results = new int[numQueries];
//     cudaMemcpy(h_results, results, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

//     // // Display results
//     std::cout << "Search results: " << std::endl;
//     for (int i = 0; i < numQueries; ++i) {
//         std::cout << "Query " << i << ": Cluster " << h_results[i] << std::endl;
//     }

//     // // Get index and print statistics
//     // auto index = rvq.get_index();
//     // auto d_index = rvq.get_gpu_index();
//     // for (int i = 0; i < numCoarseCentroids; ++i) {
//     //     for (int j = 0; j < numFineCentroids; ++j) {
//     //         std::cout << "Coarse centroid " << i << ", Fine centroid " << j
//     //                   << " has " << index[i][j].size() << " points." << std::endl;
//     //     }
//     // }
    
//     rvq.save("rvq_model.bin");

//     RVQ rvq_loaded(128, 10, 10);
//     rvq_loaded.load("rvq_model.bin");
//     rvq_loaded.search(d_queries, numQueries, results);

//     // // Copy results from device to host
//     cudaMemcpy(h_results, results, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

//     // // Display results
//     std::cout << "Search results: " << std::endl;
//     for (int i = 0; i < numQueries; ++i) {
//         std::cout << "Query " << i << ": Cluster " << h_results[i] << std::endl;
//     }

//     // Clean up
//     // delete[] trainData;
//     // delete[] queryData;

//     return 0;
// }
