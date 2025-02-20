#pragma once
#include<cuda_runtime.h>
#include "structure_on_device.cuh"

__global__ void buildReversedDetourableRoute(int* d_graph, int* d_num_of_detourable_route, int* d_reversed_graph, int* d_num_of_reversed_neighbors, int* d_num_of_reversed_detourable_route, 
                                             int offset_shift, int num_of_initial_neighbors, int num_of_iterations, int total_num_of_points){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;
    
    for (int i = 0; i < num_of_iterations; i++) {
    	int crt_point_id = i * base + b_id;
        int* crt_graph = d_graph + (crt_point_id << offset_shift);
        int* crt_num_of_detourable_route = d_num_of_detourable_route + (crt_point_id << offset_shift);
    	if (crt_point_id < total_num_of_points) {
        	for (int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int unroll_t_id = t_id + size_of_warp * j;
        	    if (unroll_t_id < num_of_initial_neighbors) {
        	        int target_point_id = crt_graph[unroll_t_id];
                    int* target_point_neighbors = d_graph + (target_point_id << offset_shift);
                    int flag = 0;
                    for( ; flag < num_of_initial_neighbors; flag++){
                        if(target_point_neighbors[flag] == crt_point_id){
                            break;
                        }
                    }
                    if(flag != num_of_initial_neighbors) continue;
                    int num_of_neighbors = atomicAdd(d_num_of_reversed_neighbors + target_point_id, 1);
                    if(num_of_neighbors < num_of_initial_neighbors){
                       (d_reversed_graph + (target_point_id * num_of_initial_neighbors))[num_of_neighbors] = crt_point_id;
                       (d_num_of_reversed_detourable_route + (target_point_id * num_of_initial_neighbors))[num_of_neighbors] = crt_num_of_detourable_route[unroll_t_id];
                    }else{
                        d_num_of_reversed_neighbors[target_point_id] = num_of_initial_neighbors;
                    }
        	    }
        	}
            
            
    	}
    }
}