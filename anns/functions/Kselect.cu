/*

    author: szr
    2024.5.11

*/

#include <limits>
#include "../utils/Utils.h"
#include "../utils/MathOperators.cuh"
#include "Kselect.cuh"

int kWarpSize = 32;
constexpr float kFloatMax = std::numeric_limits<float>::max();

template <typename T>
struct Comparator {
    __device__ static inline bool lt(T a, T b) {
        return a < b;
    }

    __device__ static inline bool gt(T a, T b) {
        return a > b;
    }
};

template <
        typename K,
        typename V,
        bool Dir,
        typename Comp,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
struct BlockSelect {
    static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
    static constexpr int kTotalWarpSortSize = NumWarpQ;

    __device__ inline BlockSelect(
            K initKVal,
            V initVVal,
            K* smemK,
            V* smemV,
            int k)
            : initK(initKVal),
              initV(initVVal),
              numVals(0),
              warpKTop(initKVal),
              sharedK(smemK),
              sharedV(smemV),
              kMinus1(k - 1) {

        // Fill the per-thread queue keys with the default value
#pragma unroll
// 将队列的threadK和threadV初始化为最大值和-1
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }

        int laneId = getLaneId();
        int warpId = threadIdx.x / kWarpSize;
        warpK = sharedK + warpId * kTotalWarpSortSize;
        warpV = sharedV + warpId * kTotalWarpSortSize;
// 将warp select的warpK和warpV初始化为最大值和-1
        // Fill warp queue (only the actual queue space is fine, not where
        // we write the per-thread queues for merging)
        for (int i = laneId; i < NumWarpQ; i += kWarpSize) {
            warpK[i] = initK;
            warpV[i] = initV;
        }

        warpFence();
    }

    __device__ inline void addThreadQ(K k, V v) {
        if (Dir ? Comp::gt(k, warpKTop) : Comp::lt(k, warpKTop)) {
            // Rotate right
#pragma unroll
            for (int i = NumThreadQ - 1; i > 0; --i) {
                threadK[i] = threadK[i - 1];
                threadV[i] = threadV[i - 1];
            }

            threadK[0] = k;
            threadV[0] = v;
            ++numVals;
        }
    }

    __device__ inline void checkThreadQ() {
        bool needSort = (numVals == NumThreadQ);

#if CUDA_VERSION >= 9000
        needSort = __any_sync(0xffffffff, needSort);
#else
        needSort = __any(needSort);
#endif

        if (!needSort) {
            // no lanes have triggered a sort
            return;
        }

        // This has a trailing warpFence
        mergeWarpQ();

        // Any top-k elements have been merged into the warp queue; we're
        // free to reset the thread queues
        numVals = 0;

#pragma unroll
        for (int i = 0; i < NumThreadQ; ++i) {
            threadK[i] = initK;
            threadV[i] = initV;
        }

        // We have to beat at least this element
        warpKTop = warpK[kMinus1];

        warpFence();
    }

    /// This function handles sorting and merging together the
    /// per-thread queues with the warp-wide queue, creating a sorted
    /// list across both
    __device__ inline void mergeWarpQ() {
        int laneId = getLaneId();

        // Sort all of the per-thread queues
        warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

        constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;
        K warpKRegisters[kNumWarpQRegisters];
        V warpVRegisters[kNumWarpQRegisters];

#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpKRegisters[i] = warpK[i * kWarpSize + laneId];
            warpVRegisters[i] = warpV[i * kWarpSize + laneId];
        }

        warpFence();

        // The warp queue is already sorted, and now that we've sorted the
        // per-thread queue, merge both sorted lists together, producing
        // one sorted list
        warpMergeAnyRegisters<
                K,
                V,
                kNumWarpQRegisters,
                NumThreadQ,
                !Dir,
                Comp,
                false>(warpKRegisters, warpVRegisters, threadK, threadV);

        // Write back out the warp queue
#pragma unroll
        for (int i = 0; i < kNumWarpQRegisters; ++i) {
            warpK[i * kWarpSize + laneId] = warpKRegisters[i];
            warpV[i * kWarpSize + laneId] = warpVRegisters[i];
        }

        warpFence();
    }

    /// WARNING: all threads in a warp must participate in this.
    /// Otherwise, you must call the constituent parts separately.
    __device__ inline void add(K k, V v) {
        addThreadQ(k, v);
        checkThreadQ();
    }

    __device__ inline void reduce() {
        // Have all warps dump and merge their queues; this will produce
        // the final per-warp results
        mergeWarpQ();

        // block-wide dep; thus far, all warps have been completely
        // independent
        __syncthreads();

        // All warp queues are contiguous in smem.
        // Now, we have kNumWarps lists of NumWarpQ elements.
        // This is a power of 2.
        FinalBlockMerge<kNumWarps, ThreadsPerBlock, K, V, NumWarpQ, Dir, Comp>::
                merge(sharedK, sharedV);

        // The block-wide merge has a trailing syncthreads
    }

    // Default element key
    const K initK;

    // Default element value
    const V initV;

    // Number of valid elements in our thread queue
    int numVals;

    // The k-th highest (Dir) or lowest (!Dir) element
    K warpKTop;

    // Thread queue values
    K threadK[NumThreadQ];
    V threadV[NumThreadQ];

    // Queues for all warps
    K* sharedK;
    V* sharedV;

    // Our warp's queue (points into sharedK/sharedV)
    // warpK[0] is highest (Dir) or lowest (!Dir)
    K* warpK;
    V* warpV;

    // This is a cached k-1 value
    int kMinus1;
};

void selectMinK(float* distances, int queryNum, int centroidNum, int k, float* outDistances, int* outIndex) {
    int grid = queryNum;
    // int block = 128;
    
    #define L2_KERNEL(BLOCK, NUM_WARP_Q, NUM_THREAD_Q)   \
    selectMinK<NUM_WARP_Q, NUM_THREAD_Q, BLOCK> \
            <<<grid, BLOCK>>>(                     \
                    distances,                         \
                    queryNum,                              \
                    centroidNum,                        \
                    outDistances,                             \
                    outIndex,                               \
                    k)
    
    if (k <= 32) {
        // const int numWarpQ = 32;
        // const int numThreadQ = 2;
        L2_KERNEL(128, 32, 2);
    } else if (k <= 64) {
        // const int numWarpQ = 64;
        // const int numThreadQ = 3;
        L2_KERNEL(128, 64, 3);
    } else if (k <= 128) {
        // const int numWarpQ = 128;
        // const int numThreadQ = 3;
        L2_KERNEL(128, 128, 3);
    }

    // selectMinK<<<grid, block>>>(numWarpQ, numThreadQ, distances, queryNum, centroidNum, outDistances, outIndex, k);

}

template<
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void selectMinK (float* distances, int queryNum, int centroidNum, float* outDistances, int* outIndex, int k) {
    // Each block handles a single row of the distances (results)
    int ThreadsPerBlock = 128;
    int kWarpSize = 32;
    int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];

    // init heap
    BlockSelect<
            dis_t,
            num_t,
            false,
            Comparator<dis_t>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kFloatMax, -1, smemK, smemV, k);

    num_t row = blockIdx.x;

    // Whole warps must participate in the selection
    int limit = roundDown(centroidNum, kWarpSize);
    int i = threadIdx.x;

    for (; i < limit; i += blockDim.x) {
        // Todo: kernel fuse
        // 传入||y||^2， 将 ||y||^2 和 -2<x,y> 距离加和
        // float v = Math<float>::add(centroidDistances[i], productDistances[row][i]);

        heap.add(distances[row * centroidNum + i], num_t(i));
    }

    if (i < centroidNum) {
        // Todo: kernel fuse
        // 传入||y||^2， 将 ||y||^2 和 -2<x,y> 距离加和
        // T v = Math<T>::add(centroidDistances[i], productDistances[row][i]);
        // heap.addThreadQ(v, IndexT(i));
        heap.addThreadQ(distances[row * centroidNum + i], num_t(i));
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outDistances[row * k + i] = smemK[i];
        outIndex[row * k + i] = idx_t(smemV[i]);
    }

}

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>

// 假设你的函数定义在这里，已包含
// ...

// 初始化一个模拟距离矩阵，其中包含从0到4999的序列，重复512次
std::vector<float> generateTestData(int queryNum, int centroidNum) {
    std::vector<float> data(queryNum * centroidNum);
    for (int q = 0; q < queryNum; ++q) {
        for (int c = 0; c < centroidNum; ++c) {
            data[q * centroidNum + c] = c % 5000; // 0到499的循环
        }
    }
    return data;
}

// 主测试函数
void testSelectMinK() {
    const int queryNum = 512;
    const int centroidNum = 5000;
    const int k = 128;

    // 初始化数据
    std::vector<float> h_distances = generateTestData(queryNum, centroidNum);
    std::vector<float> h_outDistances(queryNum * k, 0.f);
    std::vector<int> h_outIndex(queryNum * k, -1);

    // 分配GPU内存
    float* d_distances, *d_outDistances;
    int* d_outIndex;
    cudaMalloc(&d_distances, sizeof(float) * queryNum * centroidNum);
    cudaMalloc(&d_outDistances, sizeof(float) * queryNum * k);
    cudaMalloc(&d_outIndex, sizeof(int) * queryNum * k);

    // 数据传输至GPU
    cudaMemcpy(d_distances, h_distances.data(), sizeof(float) * queryNum * centroidNum, cudaMemcpyHostToDevice);

    // 调用函数
    selectMinK(d_distances, queryNum, centroidNum, k, d_outDistances, d_outIndex);

    // 结果复制回CPU
    cudaMemcpy(h_outDistances.data(), d_outDistances, sizeof(float) * queryNum * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outIndex.data(), d_outIndex, sizeof(int) * queryNum * k, cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(d_distances);
    cudaFree(d_outDistances);
    cudaFree(d_outIndex);

    // 验证结果
    // 由于直接验证具体数字可能较复杂，通常我们会检查输出的逻辑是否符合预期，例如是否为每个query的前k个最小值
    // 下面是一个简化的验证逻辑，实际上你需要根据具体输出逻辑设计验证函数
    for (int q = 0; q < queryNum; ++q) {
        std::sort(h_outDistances.begin() + q * k, h_outDistances.begin() + (q + 1) * k); // 确保排序以便验证
        for (int i = 0; i < k; ++i) {
            // 这里简化假设最小的k个值应该是从0开始，实际上你需要根据实际逻辑来验证
            assert(h_outDistances[q * k + i] == i); 
            printf("i = %d", i);
        }
    }
    std::cout << "Test passed." << std::endl;
}

int main() {
    testSelectMinK();
    return 0;
}