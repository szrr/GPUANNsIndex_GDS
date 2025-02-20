#pragma once

#include<cuda_runtime.h>
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "../graph_kernel_operation/warpselect/WarpSelect.cuh"

template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void kSubgraphMerge(array_d_t<int> *ssd_graph, array_d_t<float> *ssd_dis, size_t* d_pre_fix_of_start_loc, int* d_cluster, int* d_offset, int* d_final_graph, 
                               int* d_pre_fix_of_subgraph_size, int k, int num_of_points, int num_of_subgraph, int num_of_neighbors, int num_of_total_points, int base_id){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int num_of_blocks = gridDim.x;
    int size_of_warp = 32;
    // size_t num_of_final_neighbors = size_t(num_of_neighbors) * 2;
    size_t num_of_final_neighbors = size_t(num_of_neighbors);
    WarpSelect<float, int, false, Comparator<float>, NumWarpQ, NumThreadQ, WARP_SIZE>heap(MAX, num_of_total_points, NumWarpQ);
    __shared__ uint32_t data[size32];
    uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
    };
    for(int i = 0; i < (num_of_points + num_of_blocks - 1) / num_of_blocks; i++){
        int target_point_id = b_id + i * num_of_blocks;
        if(target_point_id < num_of_points){
            target_point_id += base_id;
            int* start_of_cluster = d_cluster + size_t(target_point_id) * size_t(k);
            int* start_of_offset = d_offset + size_t(target_point_id) * size_t(k);
            for(int l = 0; l < k; l++){
                int cluster_id = start_of_cluster[l];
                // if(b_id == 0 && target_point_id == 0){
                //     printf("%d\n",d_graph[(d_pre_fix_of_subgraph_size[cluster_id] + start_of_offset[l]) * num_of_neighbors]);
                // }
                size_t offset = d_pre_fix_of_start_loc[cluster_id] + size_t(start_of_offset[l]) * size_t(num_of_neighbors);
                for(int j = 0; j < (num_of_neighbors + size_of_warp - 1) / size_of_warp; j++){
                    size_t neighbor_offset = size_t(t_id) + size_t(j) * size_t(size_of_warp);
                    if(neighbor_offset < num_of_neighbors){
                        int neighbor_id = (*ssd_graph)[offset + neighbor_offset];
                        if(neighbor_id < num_of_total_points && !test(neighbor_id, random_number, data)){
                            heap.addThreadQ((*ssd_dis)[offset + neighbor_offset], neighbor_id);
                            add(neighbor_id, random_number, data);
                        }
                    }
                    heap.checkThreadQ();
                }
            }
            for(int l = 0; l < (size32 + size_of_warp - 1) / size_of_warp; l++){
                int temp_id = t_id + l * size_of_warp;
                if(temp_id < size32){
                    data[temp_id] = 0;
                }
            }
            int* start_of_final_graph = d_final_graph + size_t(target_point_id - base_id) * size_t(num_of_final_neighbors);
            for(int l = 0; l < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; l++){
                int offset = t_id + l * size_of_warp;
                if(offset < num_of_final_neighbors){
                    start_of_final_graph[offset] = heap.warpV[l];
                }
            }
            __syncthreads();
            heap.reset();
        }
    }
}