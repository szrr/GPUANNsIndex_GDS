/*
    Start time: 2024.4.8
    author: szr
*/

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <limits>
#include <sstream>
#include <vector>
#include <cassert>

#include "configs.h"
#include "DeviceUtils.h"
#include "GpuResources.h"

// How many streams per device we allocate by default (for multi-streaming)
constexpr int kNumStreams = 2;

GpuResources::GpuResources() {
    // 初始化CUDA设备，这里只处理单个设备
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found.");
    }
    cudaSetDevice(deviceNo); // 设置当前使用的CUDA设备为设备0

    // 创建一个默认的CUDA流
    cudaStreamCreate(&defaultStream_);
}

GpuResources::~GpuResources() {
    // 销毁cuBLAS句柄
    if (blasHandle_) {
        cublasDestroy(blasHandle_);
    }

    // 销毁CUDA流
    if (defaultStream_) {
        cudaStreamDestroy(defaultStream_);
    }
    if (asyncCopyStream_) {
        cudaStreamDestroy(asyncCopyStream_);
    }
    for (auto& stream : alternateStreams_) {
        cudaStreamDestroy(stream);
    }
    alternateStreams_.clear(); // 清除流向量

    // 释放固定内存
    if (pinnedMemAlloc_) {
        cudaFreeHost(pinnedMemAlloc_);
        pinnedMemAlloc_ = nullptr;
    }

    // 释放设备内存
    if (tempMemoryPtr_) {
        cudaFree(tempMemoryPtr_);
        tempMemoryPtr_ = nullptr;
    }

    // 清理内存分配记录
    allocations_.clear();

    // 重置头指针
    head_ = nullptr;
    end_ = nullptr;
    start_ = nullptr;
}

// 获取默认CUDA流
cudaStream_t GpuResources::getDefaultStream() const {
    return defaultStream_;
}

// 分配设备内存
void* GpuResources::allocDeviceMemory(size_t size) {
    float* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

// 释放设备内存
void GpuResources::freeDeviceMemory(void* ptr) {
    cudaFree(ptr);
}

char* GpuResources::allocTempMemory(size_t size, cudaStream_t stream) {
    auto sizeRemaining = (end_ - head_);
    if(size > sizeRemaining) {
        std::cout << "TempMemory alloc not enough" << std::endl;
        return -1;
    }
    auto rounded_size = (size + 15) & ~15;
    char* startAlloc = head_;
    char* endAlloc = head_ + rounded_size;

    MemoryRange range = {startAlloc, endAlloc};
    allocations_.push_back(range);
    
    head_ = endAlloc;

    return startAlloc;
}

void GpuResources::deallocTempMemory(char* start, cudaStream_t stream) {
    if (allocations_.empty()) {
        std::cerr << "Error: No memory to deallocate." << std::endl;
        return;
    }

    // 获取最后一个分配的记录
    MemoryRange& lastAllocation = allocations_.back();

    // 确保要释放的指针与最后一次分配的指针相匹配
    if (start != lastAllocation.start) {
        std::cerr << "Error: Deallocation order mismatch." << std::endl;
        return;
    }

    // 重置头指针到上一次分配前的位置
    head_ = lastAllocation.start;

    // 移除最后一次的分配记录
    allocations_.pop_back();
}

bool GpuResources::initializeForDevice(int device){
    if(defaultStream_ != 0){
        return;
    }

    // Create streams
    cudaStream_t defaultStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));
    defaultStream_ = defaultStream;

    cudaStream_t asyncCopyStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));
    asyncCopyStream_ = asyncCopyStream;

    std::vector<cudaStream_t> deviceStreams;
    for (int j = 0; j < kNumStreams; ++j) {
        cudaStream_t stream = 0;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        deviceStreams.push_back(stream);
    }

    alternateStreams_ = std::move(deviceStreams);

    // Create cuBLAS handle
    cublasHandle_t blasHandle = 0;
    auto blasStatus = cublasCreate(&blasHandle);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    blasHandle_ = blasHandle;

    cublasSetMathMode(
            blasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

    // initial pinned memory
    auto err = cudaHostAlloc(
            &pinnedMemAlloc_, pinnedMemSize_, cudaHostAllocDefault);

    FAISS_THROW_IF_NOT_FMT(
            err == cudaSuccess,
            "failed to cudaHostAlloc %zu bytes for CPU <-> GPU "
            "async copy buffer (error %d %s)",
            pinnedMemSize_,
            (int)err,
            cudaGetErrorString(err));
    

    // initial temporary memory
    cudaMalloc(&tempMemoryPtr_, tempMemoryTotalSize_);
    FAISS_ASSERT_FMT(
            tempMemoryPtr_,
            "could not reserve temporary memory region of size %zu",
            tempMemoryTotalSize_);
    start_ = tempMemoryPtr_ + 16;
    head_ = start_;
    end_ = tempMemoryPtr_ + allocSize_;
}