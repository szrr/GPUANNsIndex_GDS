#pragma once
#include<cuda_runtime.h>
__global__
void countDetourableRoute(int* d_graph, int* d_num_of_detourable_route, int offset_shift, int num_of_iterations, int total_num_of_points){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = 1 << offset_shift;

     for (int i = 0; i < num_of_iterations; i++) {
    	int X_id = i * base + b_id;
    	int* X_graph = d_graph + (X_id << offset_shift);
        int* crt_num_of_detourable_route = d_num_of_detourable_route + (X_id << offset_shift);
    	if (X_id < total_num_of_points) {

        	for (int j = 0; j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int X_Y_loc = t_id + size_of_warp * j;
        	    if (X_Y_loc < num_of_final_neighbors) {
        	        int Y_id = X_graph[X_Y_loc];
                    if(Y_id >= total_num_of_points){
                        crt_num_of_detourable_route[X_Y_loc] = total_num_of_points;
                    }else{
                        for(int X_Z_loc = 0; X_Z_loc < num_of_final_neighbors; X_Z_loc++){
                            if(X_Z_loc == X_Y_loc) continue;
                            int Z_id = X_graph[X_Z_loc];
                            if(Z_id >= total_num_of_points) continue;
                            int* Z_graph = d_graph + (Z_id << offset_shift);
                            for(int Z_Y_loc = 0; Z_Y_loc < num_of_final_neighbors; Z_Y_loc++){
                               if(Z_graph[Z_Y_loc] == Y_id){
                                 if(X_Z_loc < X_Y_loc && Z_Y_loc < X_Y_loc){
                                    crt_num_of_detourable_route[X_Y_loc]++;
                                 }
                                 break;
                               } 
                            }
                        }
                    }
        	    }
        	}
    	}
    }
}