/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

template <typename T>
struct Math {
    typedef T ScalarType;

    static inline __device__ T add(T a, T b) {
        return a + b;
    }

    static inline __device__ T sub(T a, T b) {
        return a - b;
    }

    static inline __device__ T mul(T a, T b) {
        return a * b;
    }

    static inline __device__ T neg(T v) {
        return -v;
    }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    static inline __device__ float reduceAdd(T v) {
        return ConvertTo<float>::to(v);
    }

    static inline __device__ bool lt(T a, T b) {
        return a < b;
    }

    static inline __device__ bool gt(T a, T b) {
        return a > b;
    }

    static inline __device__ bool eq(T a, T b) {
        return a == b;
    }

    static inline __device__ T zero() {
        return (T)0;
    }
};