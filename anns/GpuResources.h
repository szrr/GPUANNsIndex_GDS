#ifndef GPU_RESOURCES_H
#define GPU_RESOURCES_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>

class GpuResources {
public:
GpuResources(int device_id = 0);
~GpuResources();

    // 禁止复制和移动构造函数和赋值运算符，以避免潜在的资源复制问题
    GpuResources(const GpuResources&) = delete;
    GpuResources& operator=(const GpuResources&) = delete;
    GpuResources(GpuResources&&) = delete;
    GpuResources& operator=(GpuResources&&) = delete;

    void initializeForDevice(int device);

    // alloc device memory
    void* allocDeviceMemory(size_t size);

    // free device memory
    void freeDeviceMemory(void* ptr);

    // alloc temporary memory
    char* allocTempMemory(size_t size, cudaStream_t stream);

    // free temporary memory
    void freeTempMemory(void* ptr, cudaStream_t stream);

    size_t getTempMemoryAvailable() const;

    // Returns the available CPU pinned memory buffer
    std::pair<void*, size_t> getPinnedMemory();
    // Returns the stream on which we perform async CPU <-> GPU copies
    cudaStream_t getAsyncCopyStream();

    // Stream and cuBLAS handle management
    cudaStream_t getDefaultStream() const;
    cublasHandle_t getBlasHandle() const;
    std::vector<cudaStream_t> getAlternateStreams() const;

    // Device and stream synchronization
    void syncDevice();
    void syncStream(cudaStream_t stream);

private:
    struct MemoryRange {
        char* start_;
        char* end_;
    };
    int device_id;
    cudaStream_t defaultStream_ = 0; // default CUDA stream
    cudaStream_t asyncCopyStream_ = 0; // async copy stream
    std::vector<cudaStream_t> alternateStreams_; // 备选CUDA流映射
    cublasHandle_t blasHandle_;  // cuBLAS handle

    float* pinnedMemAlloc_; // pinned memory ptr
    size_t pinnedMemSize_ = 1024*1024*1024; // pinned memory size, default 1GB
    
    char* tempMemoryPtr_; //Where our temporary memory buffer is allocated
    char* start_; // Our temporary memory region; [start_, end_) is valid
    char* head_; // Stack head within [start, end)
    char* end_; // Our temporary memory region; [start_, end_) is valid
    size_t tempMemoryTotalSize_ = 1024*1024*1024; // temporary memory size, default 1GB
    std::vector<MemoryRange> allocations_;
};

#endif // GPU_RESOURCES_H