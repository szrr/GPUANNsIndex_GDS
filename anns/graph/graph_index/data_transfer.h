#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// 文件路径
const char* DATA_BLOCK_PATH = "/data/szr/rvq_data_block/rvq_reorder_graph_index.bin";
const char* CLUSTER_INDEX_PATH = "/data/szr/rvq_data_block/cluster_start_index.bin";

// 数据块参数
const size_t ELEMENT_SIZE = 772;   // 每个元素大小（字节）
const size_t TOTAL_ELEMENTS = 100000000; // 总元素数（100M）
const int NUM_CLUSTERS = 1000;     // Cluster 总数
const int CLUSTER_INDEX_SIZE = 1001; // Cluster 索引数

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Cluster 传输信息
struct ClusterData {
    void* cpu_ptr;              // CPU 内存指针
    void* gpu_ptr;              // GPU 内存指针
    bool is_transferred;        // 是否已传输到 GPU
};

// 全局 Cluster 索引表
std::vector<int> cluster_start_indices;

// 全局 CPU 数据块
char* cpu_data;

// 全局 Cluster 信息
ClusterData* cluster_info = new ClusterData[NUM_CLUSTERS](); // 初始化所有值
ClusterData* d_cluster_info;

void create_cluster_info() {
    CHECK_CUDA(cudaMalloc((void**)&d_cluster_info, NUM_CLUSTERS * sizeof(ClusterData)));
    CHECK_CUDA(cudaMemcpy(d_cluster_info, cluster_info, NUM_CLUSTERS * sizeof(ClusterData), cudaMemcpyHostToDevice));
}

// 加载 Cluster 起始索引文件
void load_cluster_indices() {
    std::ifstream file(CLUSTER_INDEX_PATH, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open cluster index file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cluster_start_indices.resize(CLUSTER_INDEX_SIZE);
    file.read(reinterpret_cast<char*>(cluster_start_indices.data()), 
              CLUSTER_INDEX_SIZE * sizeof(int));
    file.close();
}

// 加载所有数据到 CPU 内存
void load_all_data_to_cpu() {
    // 分配 CPU 内存
    // cpu_data.resize(TOTAL_ELEMENTS * ELEMENT_SIZE);
    cudaMallocHost((void**)&cpu_data, TOTAL_ELEMENTS * ELEMENT_SIZE);

    // 从数据块文件读取所有数据
    std::ifstream data_file(DATA_BLOCK_PATH, std::ios::binary);
    if (!data_file.is_open()) {
        std::cerr << "Failed to open data block file!" << std::endl;
        exit(EXIT_FAILURE);
    }
    data_file.read(cpu_data, TOTAL_ELEMENTS * ELEMENT_SIZE);
    data_file.close();
}

// 传输单个 Cluster 数据
void transfer_cluster(int cluster_id, cudaStream_t stream) {
    auto total_transfer_start = std::chrono::high_resolution_clock::now();
    
    if (cluster_id < 0 || cluster_id >= NUM_CLUSTERS) {
        std::cerr << "Invalid cluster ID: " << cluster_id << std::endl;
        return;
    }

    // 获取 Cluster 的起始和结束位置
    int start_idx = cluster_start_indices[cluster_id];
    int end_idx = cluster_start_indices[cluster_id + 1];
    size_t num_elements = end_idx - start_idx;
    
    size_t data_size = (num_elements) * ELEMENT_SIZE;

    // 分配 GPU 内存
    void* gpu_ptr;
    auto start_malloc = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMalloc(&gpu_ptr, data_size));

    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    auto end_malloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> malloc_time = end_malloc - start_malloc;
    std::cout << "cudaMalloc time: " << malloc_time.count()*1000 << " ms" << std::endl;

    // 异步传输数据到 GPU
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpyAsync(gpu_ptr, cpu_data + start_idx * ELEMENT_SIZE, 
                              data_size, cudaMemcpyHostToDevice, stream));

    // 创建传输完成事件
    CHECK_CUDA(cudaEventRecord(event, stream));

    // 标记传输完成（实际需通过事件查询）
    CHECK_CUDA(cudaEventSynchronize(event));
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> memcpy_time = end_memcpy - start_memcpy;
    std::cout << "cudaMemcpyAsync time: " << memcpy_time.count()*1000 << " ms" << std::endl;

    auto start_transfer_cluster_info = std::chrono::high_resolution_clock::now();
    // 向GPU传输cluster信息
    cluster_info[cluster_id].cpu_ptr = cpu_data + start_idx * ELEMENT_SIZE;
    cluster_info[cluster_id].gpu_ptr = gpu_ptr;
    cluster_info[cluster_id].is_transferred = true;
    CHECK_CUDA(cudaMemcpyAsync(d_cluster_info + cluster_id, cluster_info + cluster_id, 
                              sizeof(ClusterData), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaEventDestroy(event));
    auto end_transfer_cluster_info = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transfer_cluster_info_time = end_transfer_cluster_info - start_transfer_cluster_info;
    std::cout << "transfer cluster info time: " << transfer_cluster_info_time.count()*1000 << " ms" << std::endl;

    auto total_transfer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = total_transfer_end - total_transfer_start;
    std::cout << "total time: " << total_time.count() << " s" << std::endl;
    std::cout << "transfer cluster bindwidth: " << (double)data_size/1024/1024/1024/total_time.count() << " GB/s" << std::endl;
}