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

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {
#if CUDA_VERSION >= 9000
    __syncwarp();
#else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
#endif
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool AllThreads,
        bool Dir,
        typename Comp,
        bool FullMerge>
inline __device__ void blockMergeSmall(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(
            isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L <= NumThreads, "merge list size must be <= NumThreads");

    // Which pair of lists we are merging
    int mergeId = threadIdx.x / L;

    // Which thread we are within the merge
    int tid = threadIdx.x % L;

    // listK points to a region of size N * 2 * L
    listK += 2 * L * mergeId;
    listV += 2 * L * mergeId;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
    int pos = L - 1 - tid;
    int stride = 2 * tid + 1;

    if (AllThreads || (threadIdx.x < N * L)) {
        K ka = listK[pos];
        K kb = listK[pos + stride];

        bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        V va = listV[pos];
        V vb = listV[pos + stride];
        listV[pos] = swap ? vb : va;
        listV[pos + stride] = swap ? va : vb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
        int pos = 2 * tid - (tid & (stride - 1));

        if (AllThreads || (threadIdx.x < N * L)) {
            K ka = listK[pos];
            K kb = listK[pos + stride];

            bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            V va = listV[pos];
            V vb = listV[pos + stride];
            listV[pos] = swap ? vb : va;
            listV[pos + stride] = swap ? va : vb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

// Merge pairs of sorted lists larger than blockDim.x (NumThreads)
template <
        int NumThreads,
        typename K,
        typename V,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
inline __device__ void blockMergeLarge(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(L >= kWarpSize, "merge list size must be >= 32");
    static_assert(
            isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L >= NumThreads, "merge list size must be >= NumThreads");

    // For L > NumThreads, each thread has to perform more work
    // per each stride.
    constexpr int kLoopPerThread = L / NumThreads;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
#pragma unroll
    for (int loop = 0; loop < kLoopPerThread; ++loop) {
        int tid = loop * NumThreads + threadIdx.x;
        int pos = L - 1 - tid;
        int stride = 2 * tid + 1;

        K ka = listK[pos];
        K kb = listK[pos + stride];

        bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        V va = listV[pos];
        V vb = listV[pos + stride];
        listV[pos] = swap ? vb : va;
        listV[pos + stride] = swap ? va : vb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

    constexpr int kSecondLoopPerThread =
            FullMerge ? kLoopPerThread : kLoopPerThread / 2;

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int loop = 0; loop < kSecondLoopPerThread; ++loop) {
            int tid = loop * NumThreads + threadIdx.x;
            int pos = 2 * tid - (tid & (stride - 1));

            K ka = listK[pos];
            K kb = listK[pos + stride];

            bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            V va = listV[pos];
            V vb = listV[pos + stride];
            listV[pos] = swap ? vb : va;
            listV[pos + stride] = swap ? va : vb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

/// Class template to prevent static_assert from firing for
/// mixing smaller/larger than block cases
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool SmallerThanBlock,
        bool FullMerge>
struct BlockMerge {};

/// Merging lists smaller than a block
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, true, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        constexpr int kNumParallelMerges = NumThreads / L;
        constexpr int kNumIterations = N / kNumParallelMerges;

        static_assert(L <= NumThreads, "list must be <= NumThreads");
        static_assert(
                (N < kNumParallelMerges) ||
                        (kNumIterations * kNumParallelMerges == N),
                "improper selection of N and L");

        if (N < kNumParallelMerges) {
            // We only need L threads per each list to perform the merge
            blockMergeSmall<
                    NumThreads,
                    K,
                    V,
                    N,
                    L,
                    false,
                    Dir,
                    Comp,
                    FullMerge>(listK, listV);
        } else {
            // All threads participate
#pragma unroll
            for (int i = 0; i < kNumIterations; ++i) {
                int start = i * kNumParallelMerges * 2 * L;

                blockMergeSmall<
                        NumThreads,
                        K,
                        V,
                        N,
                        L,
                        true,
                        Dir,
                        Comp,
                        FullMerge>(listK + start, listV + start);
            }
        }
    }
};

/// Merging lists larger than a block
template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, Comp, false, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        // Each pair of lists is merged sequentially
#pragma unroll
        for (int i = 0; i < N; ++i) {
            int start = i * 2 * L;

            blockMergeLarge<NumThreads, K, V, L, Dir, Comp, FullMerge>(
                    listK + start, listV + start);
        }
    }
};

template <
        int NumThreads,
        typename K,
        typename V,
        int N,
        int L,
        bool Dir,
        typename Comp,
        bool FullMerge = true>
inline __device__ void blockMerge(K* listK, V* listV) {
    constexpr bool kSmallerThanBlock = (L <= NumThreads);

    BlockMerge<
            NumThreads,
            K,
            V,
            N,
            L,
            Dir,
            Comp,
            kSmallerThanBlock,
            FullMerge>::merge(listK, listV);
}

template <
        typename K,
        typename V,
        int L,
        bool Dir,
        typename Comp,
        bool IsBitonic>
inline __device__ void warpBitonicMergeLE16(K& k, V& v) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(L <= kWarpSize / 2, "merge list size must be <= 16");

    int laneId = getLaneId();

    if (!IsBitonic) {
        // Reverse the first comparison stage.
        // For example, merging a list of size 8 has the exchanges:
        // 0 <-> 15, 1 <-> 14, ...
        K otherK = shfl_xor(k, 2 * L - 1);
        V otherV = shfl_xor(v, 2 * L - 1);

        // Whether we are the lesser thread in the exchange
        bool small = !(laneId & L);

        if (Dir) {
            // See the comment above how performing both of these
            // comparisons in the warp seems to win out over the
            // alternatives in practice
            bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
            assign(s, k, otherK);
            assign(s, v, otherV);

        } else {
            bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
            assign(s, k, otherK);
            assign(s, v, otherV);
        }
    }

#pragma unroll
    for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
        K otherK = shfl_xor(k, stride);
        V otherV = shfl_xor(v, stride);

        // Whether we are the lesser thread in the exchange
        bool small = !(laneId & stride);

        if (Dir) {
            bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
            assign(s, k, otherK);
            assign(s, v, otherV);

        } else {
            bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
            assign(s, k, otherK);
            assign(s, v, otherV);
        }
    }
}

// Template for performing a bitonic merge of an arbitrary set of
// registers
template <
        typename K,
        typename V,
        int N,
        bool Dir,
        typename Comp,
        bool Low,
        bool Pow2>
struct BitonicMergeStep {};

//
// Power-of-2 merge specialization
//

// All merges eventually call this
template <typename K, typename V, bool Dir, typename Comp, bool Low>
struct BitonicMergeStep<K, V, 1, Dir, Comp, Low, true> {
    static inline __device__ void merge(K k[1], V v[1]) {
        // Use warp shuffles
        warpBitonicMergeLE16<K, V, 16, Dir, Comp, true>(k[0], v[0]);
    }
};

template <typename K, typename V, int N, bool Dir, typename Comp, bool Low>
struct BitonicMergeStep<K, V, N, Dir, Comp, Low, true> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(isPowerOf2(N), "must be power of 2");
        static_assert(N > 1, "must be N > 1");

#pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            K& ka = k[i];
            V& va = v[i];

            K& kb = k[i + N / 2];
            V& vb = v[i + N / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
            swap(s, va, vb);
        }

        {
            K newK[N / 2];
            V newV[N / 2];

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                newK[i] = k[i];
                newV[i] = v[i];
            }

            BitonicMergeStep<K, V, N / 2, Dir, Comp, true, true>::merge(
                    newK, newV);

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                k[i] = newK[i];
                v[i] = newV[i];
            }
        }

        {
            K newK[N / 2];
            V newV[N / 2];

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                newK[i] = k[i + N / 2];
                newV[i] = v[i + N / 2];
            }

            BitonicMergeStep<K, V, N / 2, Dir, Comp, false, true>::merge(
                    newK, newV);

#pragma unroll
            for (int i = 0; i < N / 2; ++i) {
                k[i + N / 2] = newK[i];
                v[i + N / 2] = newV[i];
            }
        }
    }
};

//
// Non-power-of-2 merge specialization
//

// Low recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStep<K, V, N, Dir, Comp, true, false> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(!isPowerOf2(N), "must be non-power-of-2");
        static_assert(N >= 3, "must be N >= 3");

        constexpr int kNextHighestPowerOf2 = nextHighestPowerOf2(N);

#pragma unroll
        for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
            K& ka = k[i];
            V& va = v[i];

            K& kb = k[i + kNextHighestPowerOf2 / 2];
            V& vb = v[i + kNextHighestPowerOf2 / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
            swap(s, va, vb);
        }

        constexpr int kLowSize = N - kNextHighestPowerOf2 / 2;
        constexpr int kHighSize = kNextHighestPowerOf2 / 2;
        {
            K newK[kLowSize];
            V newV[kLowSize];

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                newK[i] = k[i];
                newV[i] = v[i];
            }

            constexpr bool kLowIsPowerOf2 =
                    isPowerOf2(N - kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kLowIsPowerOf2 = isPowerOf2(kLowSize);
            BitonicMergeStep<
                    K,
                    V,
                    kLowSize,
                    Dir,
                    Comp,
                    true, // low
                    kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                k[i] = newK[i];
                v[i] = newV[i];
            }
        }

        {
            K newK[kHighSize];
            V newV[kHighSize];

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                newK[i] = k[i + kLowSize];
                newV[i] = v[i + kLowSize];
            }

            constexpr bool kHighIsPowerOf2 =
                    isPowerOf2(kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kHighIsPowerOf2 =
            //      isPowerOf2(kHighSize);
            BitonicMergeStep<
                    K,
                    V,
                    kHighSize,
                    Dir,
                    Comp,
                    false, // high
                    kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                k[i + kLowSize] = newK[i];
                v[i + kLowSize] = newV[i];
            }
        }
    }
};

// High recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStep<K, V, N, Dir, Comp, false, false> {
    static inline __device__ void merge(K k[N], V v[N]) {
        static_assert(!isPowerOf2(N), "must be non-power-of-2");
        static_assert(N >= 3, "must be N >= 3");

        constexpr int kNextHighestPowerOf2 = nextHighestPowerOf2(N);

#pragma unroll
        for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
            K& ka = k[i];
            V& va = v[i];

            K& kb = k[i + kNextHighestPowerOf2 / 2];
            V& vb = v[i + kNextHighestPowerOf2 / 2];

            bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            swap(s, ka, kb);
            swap(s, va, vb);
        }

        constexpr int kLowSize = kNextHighestPowerOf2 / 2;
        constexpr int kHighSize = N - kNextHighestPowerOf2 / 2;
        {
            K newK[kLowSize];
            V newV[kLowSize];

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                newK[i] = k[i];
                newV[i] = v[i];
            }

            constexpr bool kLowIsPowerOf2 =
                    isPowerOf2(kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kLowIsPowerOf2 = isPowerOf2(kLowSize);
            BitonicMergeStep<
                    K,
                    V,
                    kLowSize,
                    Dir,
                    Comp,
                    true, // low
                    kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kLowSize; ++i) {
                k[i] = newK[i];
                v[i] = newV[i];
            }
        }

        {
            K newK[kHighSize];
            V newV[kHighSize];

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                newK[i] = k[i + kLowSize];
                newV[i] = v[i + kLowSize];
            }

            constexpr bool kHighIsPowerOf2 =
                    isPowerOf2(N - kNextHighestPowerOf2 / 2);
            // FIXME: compiler doesn't like this expression? compiler bug?
            //      constexpr bool kHighIsPowerOf2 =
            //      isPowerOf2(kHighSize);
            BitonicMergeStep<
                    K,
                    V,
                    kHighSize,
                    Dir,
                    Comp,
                    false, // high
                    kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
            for (int i = 0; i < kHighSize; ++i) {
                k[i + kLowSize] = newK[i];
                v[i + kLowSize] = newV[i];
            }
        }
    }
};

/// Merges two sets of registers across the warp of any size;
/// i.e., merges a sorted k/v list of size kWarpSize * N1 with a
/// sorted k/v list of size kWarpSize * N2, where N1 and N2 are any
/// value >= 1
template <
        typename K,
        typename V,
        int N1,
        int N2,
        bool Dir,
        typename Comp,
        bool FullMerge = true>
inline __device__ void warpMergeAnyRegisters(
        K k1[N1],
        V v1[N1],
        K k2[N2],
        V v2[N2]) {
    constexpr int kSmallestN = N1 < N2 ? N1 : N2;

#pragma unroll
    for (int i = 0; i < kSmallestN; ++i) {
        K& ka = k1[N1 - 1 - i];
        V& va = v1[N1 - 1 - i];

        K& kb = k2[i];
        V& vb = v2[i];

        K otherKa;
        V otherVa;

        if (FullMerge) {
            // We need the other values
            otherKa = shfl_xor(ka, kWarpSize - 1);
            otherVa = shfl_xor(va, kWarpSize - 1);
        }

        K otherKb = shfl_xor(kb, kWarpSize - 1);
        V otherVb = shfl_xor(vb, kWarpSize - 1);

        // ka is always first in the list, so we needn't use our lane
        // in this comparison
        bool swapa = Dir ? Comp::gt(ka, otherKb) : Comp::lt(ka, otherKb);
        assign(swapa, ka, otherKb);
        assign(swapa, va, otherVb);

        // kb is always second in the list, so we needn't use our lane
        // in this comparison
        if (FullMerge) {
            bool swapb = Dir ? Comp::lt(kb, otherKa) : Comp::gt(kb, otherKa);
            assign(swapb, kb, otherKa);
            assign(swapb, vb, otherVa);

        } else {
            // We don't care about updating elements in the second list
        }
    }

    BitonicMergeStep<K, V, N1, Dir, Comp, true, isPowerOf2(N1)>::merge(
            k1, v1);
    if (FullMerge) {
        // Only if we care about N2 do we need to bother merging it fully
        BitonicMergeStep<K, V, N2, Dir, Comp, false, isPowerOf2(N2)>::
                merge(k2, v2);
    }
}

// Recursive template that uses the above bitonic merge to perform a
// bitonic sort
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicSortStep {
    static inline __device__ void sort(K k[N], V v[N]) {
        static_assert(N > 1, "did not hit specialized case");

        // Sort recursively
        constexpr int kSizeA = N / 2;
        constexpr int kSizeB = N - kSizeA;

        K aK[kSizeA];
        V aV[kSizeA];

#pragma unroll
        for (int i = 0; i < kSizeA; ++i) {
            aK[i] = k[i];
            aV[i] = v[i];
        }

        BitonicSortStep<K, V, kSizeA, Dir, Comp>::sort(aK, aV);

        K bK[kSizeB];
        V bV[kSizeB];

#pragma unroll
        for (int i = 0; i < kSizeB; ++i) {
            bK[i] = k[i + kSizeA];
            bV[i] = v[i + kSizeA];
        }

        BitonicSortStep<K, V, kSizeB, Dir, Comp>::sort(bK, bV);

        // Merge halves
        warpMergeAnyRegisters<K, V, kSizeA, kSizeB, Dir, Comp>(aK, aV, bK, bV);

#pragma unroll
        for (int i = 0; i < kSizeA; ++i) {
            k[i] = aK[i];
            v[i] = aV[i];
        }

#pragma unroll
        for (int i = 0; i < kSizeB; ++i) {
            k[i + kSizeA] = bK[i];
            v[i + kSizeA] = bV[i];
        }
    }
};

// Single warp (N == 1) sorting specialization
template <typename K, typename V, bool Dir, typename Comp>
struct BitonicSortStep<K, V, 1, Dir, Comp> {
    static inline __device__ void sort(K k[1], V v[1]) {
        // Update this code if this changes
        // should go from 1 -> kWarpSize in multiples of 2
        // static_assert(kWarpSize == 32, "unexpected warp size");

        warpBitonicMergeLE16<K, V, 1, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 2, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 4, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 8, Dir, Comp, false>(k[0], v[0]);
        warpBitonicMergeLE16<K, V, 16, Dir, Comp, false>(k[0], v[0]);
    }
};

/// Sort a list of kWarpSize * N elements in registers, where N is an
/// arbitrary >= 1
template <typename K, typename V, int N, bool Dir, typename Comp>
inline __device__ void warpSortAnyRegisters(K k[N], V v[N]) {
    BitonicSortStep<K, V, N, Dir, Comp>::sort(k, v);
}

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