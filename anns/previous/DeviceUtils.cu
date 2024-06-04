#include <cuda_profiler_api.h>
#include "DeviceUtils.h"
#include <mutex>
#include <unordered_map>

int getDeviceForAddress(const void* p) {
    if (!p) {
        return -1;
    }

    cudaPointerAttributes att;
    cudaError_t err = cudaPointerGetAttributes(&att, p);

    if (err == cudaErrorInvalidValue) {
        // Make sure the current thread error status has been reset
        err = cudaGetLastError();
        return -1;
    }

    // FIXME: what to use for managed memory?
    if (att.type == cudaMemoryTypeDevice) {
        return att.device;
    } else {
        return -1;
    }
}

// Helper function to get device properties and cache them
const cudaDeviceProp& getDeviceProperties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaDeviceProp> properties;

    std::lock_guard<std::mutex> guard(mutex);

    auto it = properties.find(device);
    if (it == properties.end()) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        properties[device] = prop;
        it = properties.find(device);
    }

    return it->second;
}

// Function to get the maximum number of threads per block for a given device
int getMaxThreads(int device) {
    return getDeviceProperties(device).maxThreadsPerBlock;
}