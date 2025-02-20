#pragma once

#include<cuda_runtime.h>

__global__ void populate_chunk_distances(size_t num_of_query, size_t dim_of_query, size_t num_of_chunks, float *d_query_vec, float *d_dist_vec, 
                                        float *d_centroid, uint32_t *chunk_offsets, uint8_t *d_pq_data, float *d_pq_tables){
    size_t t_id = size_t(threadIdx.x);
    size_t b_id = size_t(blockIdx.x);
    size_t size_of_warp = 32;
    float *crt_query_vec = d_query_vec + dim_of_query * b_id;
    float* crt_chunk_dists = d_dist_vec + b_id * 256 * num_of_chunks;
    for(size_t i = 0; i < (dim_of_query + size_of_warp - 1) / size_of_warp; i++){
        size_t idx = t_id + i * size_of_warp;
        if(idx < dim_of_query){
            crt_query_vec[idx] -= d_centroid[idx];
        }
    }
    __syncthreads();
    for (size_t chunk = 0; chunk < num_of_chunks; chunk++){
        float *chunk_dists = crt_chunk_dists + (256 * chunk);
        for (size_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++)
        {
            const float *centers_dim_vec = d_pq_tables + (256 * j);
            for (size_t i = 0; i < (256 + size_of_warp - 1) / size_of_warp; i++)
            {
                size_t idx = t_id + i * size_of_warp;
                if(idx < 256){
                    double diff = centers_dim_vec[idx] - (crt_query_vec[j]);
                    chunk_dists[idx] += (float)(diff * diff);
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
    // if(b_id == 0 && t_id == 0){
    //     for(int i = 0; i < 256 * 1; i++){
    //         printf("%f ", crt_chunk_dists[i]);
    //     }
    //     printf("\n");
    // }
}