#pragma once
#include<cuda_runtime.h>
#include "warpselect/WarpSelect.cuh"
#include "structure_on_device.cuh"
template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void sortDetourableRoute(int* d_graph, int* d_num_of_detourable_route, int offset_shift, int num_of_initial_neighbors, int num_of_iterations, 
                                    int total_num_of_points){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = 1 << offset_shift;
    WarpSelect<int, int, false, Comparator<int>, NumWarpQ, NumThreadQ, WARP_SIZE>heap(MAX, total_num_of_points, (1 << offset_shift));
    for (int i = 0; i < num_of_iterations; i++) {
    	int crt_point_id = i * base + b_id;
    	int* crt_graph = d_graph + (crt_point_id << offset_shift);
		int* crt_num_of_detourable_route = d_num_of_detourable_route + (crt_point_id << offset_shift);
    	if (crt_point_id < total_num_of_points) {
        	for (int j = 0; j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int unroll_t_id = t_id + size_of_warp * j;
        	    if (unroll_t_id < num_of_final_neighbors) {
        	        heap.addThreadQ(crt_num_of_detourable_route[unroll_t_id], crt_graph[unroll_t_id]);
        	    }
        	}
            heap.reduce();
            for (int l = 0; l < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; l++) {
                int unrollt_id = t_id + size_of_warp * l;
                if (unrollt_id < num_of_initial_neighbors) {
                    crt_graph[unrollt_id] = heap.warpV[l];
                    crt_num_of_detourable_route[unrollt_id] = heap.warpK[l];
                }
            }
    	}
        heap.reset();
    }
}