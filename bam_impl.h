#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <functional>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <getopt.h>
#include <limits>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <util.h>
#include <page_cache.h>

const uint64_t KB = 1024LL;
const uint64_t MB = 1024LL * 1024;
const uint64_t GB = 1024LL * 1024 * 1024;


enum class dataset {
    SIFT1M = 0,
    SIFT10M = 1,
    SIFT1B = 2,
    GIST1M = 3
};


struct Parameters
{
    uint32_t        cudaDevice;
    uint64_t        cudaDeviceId;
    const char*     blockDevicePath;
    const char*     controllerPath;
    uint64_t        controllerId;
    uint32_t        adapter;
    uint32_t        segmentId;
    uint32_t        nvmNamespace;
    bool            doubleBuffered;
    size_t          numPages;
    size_t          startBlock;
    bool            stats;
    size_t          ofileoffset; 
    size_t          type; 
    size_t          numThreads;
    size_t          n_items;
    size_t          warp_size;
    size_t          divs;
    uint32_t        domain;
    uint32_t        bus;
    uint32_t        devfn;
    uint32_t n_ctrls;
    size_t blkSize;
    size_t queueDepth;
    size_t numQueues;
    size_t pageSize;
    uint64_t maxPageCacheSize;
    uint64_t ssdtype;
    uint64_t dim;
    uint64_t len_item;
    dataset ds;
    std::string binpath;

    Parameters();

};



Parameters::Parameters()
{
    cudaDevice = 0;
    cudaDeviceId = 0;
    blockDevicePath = nullptr;
    controllerPath = nullptr;
    controllerId = 0;
    adapter = 0;
    segmentId = 0;
    nvmNamespace = 1;
    doubleBuffered = false;
    numPages = 1024;
    startBlock = 0;
    stats = false;
    ofileoffset = 0;
    type = 1;
    numThreads = 1024;
    blkSize = 64;
    domain = 0;
    bus = 0;
    devfn = 0;
    n_ctrls = 1;
    queueDepth = 1024*4;
    numQueues = 16;
    pageSize = 65536;       // 64 KB

    /*---------------------------------------------*/
    ds = dataset::SIFT1M;
    switch (ds)
    {
    case dataset::SIFT1M:
        n_items = 1000000;
        dim = 128;
        len_item = 256;
        binpath = "/home/wind/datasets/sift/sift_base.fvecs";
        // binpath = "/mnt/data2/szr/dataset/sift/sift_base.fvecs";
        maxPageCacheSize = 200*MB;
        break;
    case dataset::SIFT10M:
        n_items = 10000000;
        dim = 128;
        len_item = 256;
        binpath = "/home/wind/datasets/sift10m/sift10m_base.fvecs";
        maxPageCacheSize = 2*GB;
        break;
    case dataset::SIFT1B:
        n_items = 1000000000;
        dim = 128;
        len_item = 256;
        binpath = "/home/wind/datasets/sift10m/sift10m_base.fvecs";
        maxPageCacheSize = 16*GB;
        break;
    case dataset::GIST1M:
        n_items = 1000000;
        dim = 960;
        len_item = 1024;
        binpath = "/home/wind/datasets/sift10m/sift10m_base.fvecs";
        break;
    default:
        printf("INVALID VAL OF DATASET!\n");
        break;
    } 

    // maxPageCacheSize = 17179869184; // 16 GB
    ssdtype = 0;

    warp_size = 128;
    divs = 8;

}

