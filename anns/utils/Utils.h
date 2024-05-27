/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

// allow usage for non-CUDA files
#ifndef __host__
#define __host__
#define __device__
#endif

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
    return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
    return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
    return divUp(a, b) * b;
}

template <class T>
constexpr __host__ __device__ T pow(T n, T power) {
    return (power > 0 ? n * pow(n, power - 1) : 1);
}

template <class T>
constexpr __host__ __device__ T pow2(T n) {
    return pow(2, (T)n);
}

static_assert(pow2(8) == 256, "pow2");

template <typename T>
constexpr __host__ __device__ int log2(T n, int p = 0) {
    return (n <= 1) ? p : log2(n / 2, p + 1);
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
    return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
    return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));
}