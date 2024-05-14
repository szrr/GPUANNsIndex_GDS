/*

    author: szr
    2024.5.11

*/

#include <limits>
#include "../utils/Utils.h"
#include "../utils/MathOperators.cuh"

int kWarpSize = 32;
constexpr float kFloatMax = std::numeric_limits<float>::max();

void selectMinK(float* distances, int queryNum, int centroidNum, int k, float* outDistances, int* outIndex) {
    int grid = queryNum;
    int block = 128;
    int numWarpQ, numThreadQ;
    if (k <= 32) {
        numWarpQ = 32;
        numThreadQ = 2;
    } else if (k <= 64) {
        numWarpQ = 64;
        numThreadQ = 3;
    } else if (k <= 128) {
        numWarpQ = 128;
        numThreadQ = 3;
    }

    selectMinK<<<grid, block>>>(numWarpQ, numThreadQ, distances, queryNum, centroidNum, outDistances, outIndex);

}

__global__ void selectMinK (int numWarpQ, int numThreadQ, float* distances, int queryNum, int centroidNum, float* outDistances, int* outIndex) {
    // Each block handles a single row of the distances (results)
    int ThreadsPerBlock = 128;
    int kWarpSize = 32;
    int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ float smemK[kNumWarps * numWarpQ];
    __shared__ int smemV[kNumWarps * numWarpQ];

    // init heap
    // BlockSelect<
    //         T,
    //         IndexT,
    //         false,
    //         Comparator<T>,
    //         NumWarpQ,
    //         NumThreadQ,
    //         ThreadsPerBlock>
    //         heap(initK, -1, smemK, smemV, k);

    int row = blockIdx.x;

    // Whole warps must participate in the selection
    int limit = roundDown(centroidNum, kWarpSize);
    int i = threadIdx.x;

    for (; i < limit; i += blockDim.x) {
        // 将 ||y||^2 和 -2<x,y> 距离加和
        // float v = Math<float>::add(centroidDistances[i], productDistances[row][i]);

        // 
        // heap.add(v, IndexT(i));
    }


}