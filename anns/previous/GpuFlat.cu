/*
    Start time: 2024.4.8
    author: szr
*/

#include <iostream>
#include "GpuFlat.h"
#include "configs.h"
#include "GpuResources.h"
#include "DeviceUtils.h"
#include "L2Norm.cuh"

inline void chooseTileSize(
        int64_t numQueries,
        int64_t numCentroids,
        int dim,
        int64_t elementSize,
        size_t tempMemAvailable,
        int64_t& tileRows,
        int64_t& tileCols) {
    // The matrix multiplication should be large enough to be efficient, but if
    // it is too large, we seem to lose efficiency as opposed to
    // double-streaming. Each tile size here defines 1/2 of the memory use due
    // to double streaming. We ignore available temporary memory, as that is
    // adjusted independently by the user and can thus meet these requirements
    // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
    // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
    auto totalMem = getDeviceProperties(deviceNo).totalGlobalMem;

    idx_t targetUsage = 0;

    if (totalMem <= ((size_t)4) * 1024 * 1024 * 1024) {
        targetUsage = 512 * 1024 * 1024;
    } else if (totalMem <= ((size_t)8) * 1024 * 1024 * 1024) {
        targetUsage = 768 * 1024 * 1024;
    } else {
        targetUsage = 1024 * 1024 * 1024;
    }

    targetUsage /= 2 * elementSize;

    // 512 seems to be a batch size sweetspot for float32.
    // If we are on float16, increase to 512.
    // If the k size (vec dim) of the matrix multiplication is small (<= 32),
    // increase to 1024.
    idx_t preferredTileRows = 512;
    if (dim <= 32) {
        preferredTileRows = 1024;
    }

    tileRows = std::min(preferredTileRows, numQueries);

    // tileCols is the remainder size
    tileCols = std::min(targetUsage / preferredTileRows, numCentroids);
}

GpuFlat::GpuFlat(GpuResources* resource, int dims, metricType metric)
    : resources_(resource), dims_(dims), metric_(metric)
    {
        resources_->initializeForDevice(deviceNo);
    }

void GpuFlat::add(idx_t n, const float* x){
    // extend originVector_ size
    size_t extendSize = num_ + n;
    float* newArray = static_cast<float*>(resources_->allocDeviceMemory(extendSize * sizeof(float)));
    auto stream = resources_->getDefaultStream();

    // Copy over any old data
    if(num_ > 0 && originVectors_ != nullptr){
        CUDA_VERIFY(cudaMemcpyAsync(
                newArray,
                originVectors_,
                num_,
                cudaMemcpyDeviceToDevice,
                stream));
        resources_->freeDeviceMemory(originVectors_);
    }

    // add x to originVector_
    int dev = getDeviceForAddress(x);
    if (dev == -1) {
        CUDA_VERIFY(cudaMemcpyAsync(
                newArray + num_,
                x,
                n * sizeof(float),
                cudaMemcpyHostToDevice,
                stream));
    } else {
        CUDA_VERIFY(cudaMemcpyAsync(
                newArray + num_,
                x,
                n * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
    }

    // 确保所有异步操作完成
    CUDA_VERIFY(cudaStreamSynchronize(stream));

    // update GpuFlat paramters
    num_ += n;
    originVectors_ = newArray;

    //precompute L2 norms

}

void GpuFlat::search(int64_t n, const float* x, int k, 
        float* distance, int64_t* labels){
    auto stream = resources_->getDefaultStream();
    if (metric_ == L2){
        // cal L2 distance
        auto numVectors = num_;
        auto numQuery = n;
        auto dim = dims_;

        // assert the dims and nums of vector match the distance and labels

        // L2 = ||q||^2 + 2<q, v> + ||v||^2
        // If ||v||^2 is not pre-computed, calculate it
        float* vNorms = (float*)(resources_->allocTempMemory(num_ * sizeof(float), stream));
        runL2Norm(originVectors_, num_, dims_, vNorms, true, deviceNo, stream);

        // calculate ||v||^2
        float* qNorms = (float*)(resources_->allocTempMemory(numQuery * sizeof(float), stream));

        // set tile size
        int64_t tileRows = 0;
        int64_t tileCols = 0;
        chooseTileSize(
                numQuery,
                numVectors,
                dim,
                sizeof(float),
                resources_->getTempMemoryAvailable(),
                tileRows,
                tileCols);

        //Todo: Distance.cu 200行 开始的内容

        // calculate 2<q, v>


    }
}