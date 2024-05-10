#include <algorithm>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include "L2Norm.cuh"
#include "DeviceUtils.h"

constexpr int kWarpSize = 32;
constexpr int rowTileSize = 8;

int divUp(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

// warpReduceAll 函数，用于在一个 warp 内进行归约
template<typename T, int ReduceWidth = kWarpSize>
__device__ T warpReduceAll(T val) {
    // 使用 full mask 以确保所有线程在同步
    unsigned int full_mask = 0xffffffff;

    for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
        T other_val = __shfl_xor_sync(full_mask, val, mask);
        val = add()(val, other_val);
    }

    return val;
}

__global__ void L2Norm(float* vectors, size_t numVec, int dims, float* res, bool NormLoop, bool NormSquared) {
    extern __shared__ char smemByte[]; // #warps * RowTileSize elements
    float* smem = (float*)smemByte;

    // these are fine to be int (just based on block dimensions)
    int numWarps = divUp(blockDim.x, kWarpSize);
    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;

    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int64_t rowStart = int64_t(blockIdx.x) * rowTileSize;
    // accumulate in f32
    float rowNorm[rowTileSize];

    if (lastRowTile) {
        // We are handling the very end of the input matrix rows
        for (int64_t row = 0; row < numVec - rowStart; ++row) {
            if (NormLoop) {
                rowNorm[0] = 0;

                for (int64_t col = threadIdx.x; col < dims; col += blockDim.x) {

                    //Todo: add float4 calculation
                    float val = vectors[(rowStart + row) * dims + col];
                    val = val * val;
                    rowNorm[0] = rowNorm[0] + val;
                }
            } else {
                float val = vectors[(rowStart + row) * dims + threadIdx.x];
                val = val * val;
                rowNorm[0] = rowNorm[0] + val;
            }

            rowNorm[0] = warpReduceAll(rowNorm[0]);
            if (laneId == 0) {
                smem[row * numWarps + warpId] = rowNorm[0];
            }
        }
    } else {
        // We are guaranteed that all RowTileSize rows are available in
        // [rowStart, rowStart + RowTileSize)

        if (NormLoop) {
            // A single block of threads is not big enough to span each
            // vector
            float tmp[rowTileSize];

#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                rowNorm[row] = 0;
            }

            for (int64_t col = threadIdx.x; col < dims; col += blockDim.x) {
#pragma unroll
                for (int row = 0; row < rowTileSize; ++row) {
                    tmp[row] = vectors[(rowStart + row) * dims + col];
                }

#pragma unroll
                for (int row = 0; row < rowTileSize; ++row) {
                    tmp[row] = tmp[row] * tmp[row];
                }

#pragma unroll
                for (int row = 0; row < rowTileSize; ++row) {
                    rowNorm[row] =
                            rowNorm[row] + tmp[row];
                }
            }
        } else {
            float tmp[rowTileSize];

            // A block of threads is the exact size of the vector
#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                tmp[row] = vectors[(rowStart + row) * dims + threadIdx.x];
            }

#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                tmp[row] = tmp[row] * tmp[row];
            }

#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                rowNorm[row] = tmp[row];
            }
        }

        // Sum up all parts in each warp
#pragma unroll
        for (int row = 0; row < rowTileSize; ++row) {
            rowNorm[row] = warpReduceAll(rowNorm[row]);
        }

        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                smem[row * numWarps + warpId] = rowNorm[row];
            }
        }
    }

    __syncthreads();

    // Sum across warps
    if (warpId == 0) {
#pragma unroll
        for (int row = 0; row < rowTileSize; ++row) {
            rowNorm[row] =
                    laneId < numWarps ? smem[row * numWarps + laneId] : 0;
        }

#pragma unroll
        for (int row = 0; row < rowTileSize; ++row) {
            rowNorm[row] = warpReduceAll(rowNorm[row]);
        }

        // Write out answer
        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < rowTileSize; ++row) {
                int outCol = rowStart + row;

                if (lastRowTile) {
                    if (outCol < numVec) {
                        res[outCol] = NormSquared
                                ? rowNorm[row]
                                : sqrtf(rowNorm[row]);
                    }
                } else {
                    res[outCol] = NormSquared
                            ? rowNorm[row]
                            : sqrtf(rowNorm[row]);
                }
            }
        }
    }
}

void runL2Norm(float* vectors, size_t numVec, int dims, float* res, bool normSquared, int deviceNo, cudaStream_t stream) {
    int maxThreads = getMaxThreads(deviceNo);

    // Todo: load using the vectorized type

    auto dim = dims;
    bool normLoop = dim > maxThreads;
    auto numThreads = std::min(dim, maxThreads);

    auto grid = dim3(divUp(numVec, rowTileSize));
    auto block = dim3(numThreads);

    auto smem = sizeof(float) * rowTileSize *
            divUp(numThreads, kWarpSize);

    if (normLoop) {
        if (normSquared) {
            L2Norm<<<grid, block, smem, stream>>>(vectors, numVec, dims, res, true, true);
        } else {
            L2Norm<<<grid, block, smem, stream>>>(vectors, numVec, dims, res, true, false);
        }
    } else {
        if (normSquared) {
            L2Norm<<<grid, block, smem, stream>>>(vectors, numVec, dims, res, false, true);
        } else {
            L2Norm<<<grid, block, smem, stream>>>(vectors, numVec, dims, res, false, false);
        }
    }

    CUDA_VERIFY(cudaGetLastError());
}