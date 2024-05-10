/*
    Start time: 2024.4.8
    author: szr
*/

#include <iostream>
#include "configs.h"
#include "GpuResources.h"

// using idx_t = int64_t;

class GpuFlat;

class GpuFlat{
    public:

    //construct
    GpuFlat(
        GpuResources* resource,
        int dims,
        metricType metric
    );

    ~GpuFlat();

    void add(int64_t n, const float* x);
    // void train(int n, const float* x);
    void search(int64_t numQuery, const float* x, int k, float* distance, int64_t* labels);

    private:
    
    int dims_;
    int64_t num_ = 0;
    float* originVectors_;
    metricType metric_;
    GpuResources* resources_;
    float* norms;
};