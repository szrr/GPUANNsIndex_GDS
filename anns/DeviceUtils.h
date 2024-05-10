#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// How many streams per device we allocate by default (for multi-streaming)
constexpr int NumStreams = 2;

// Use 1024 MiB of pinned memory for async CPU <-> GPU copies by default
constexpr size_t DefaultPinnedMemoryAllocation = (size_t)1024 * 1024 * 1024;

// Default 8GiB temporary memory allocation for 48GiB A6000
constexpr size_t DefaultTempMem = (size_t)8 * 1024 * 1024 * 1024;


/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                      \
    do {                                    \
        auto err__ = (X);                   \
        FAISS_ASSERT_FMT(                   \
                err__ == cudaSuccess,       \
                "CUDA error %d %s",         \
                (int)err__,                 \
                cudaGetErrorString(err__)); \
    } while (0)

#define FAISS_ASSERT_FMT(X, FMT, ...)                    \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__,                            \
                    __VA_ARGS__);                        \
            abort();                                     \
        }                                                \
    } while (false)

#define FAISS_THROW_IF_NOT_FMT(X, FMT, ...)                               \
    do {                                                                  \
        if (!(X)) {                                                       \
            FAISS_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                 \
    } while (false)

const cudaDeviceProp& getDeviceProperties(int device);

int getMaxThreads(int device);