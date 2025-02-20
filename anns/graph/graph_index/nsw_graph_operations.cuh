#pragma once
#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <byteswap.h>


#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cooperative_groups.h>
#include "../graph_kernel_operation/cagra_bam_pq.cuh"
#include "../graph_kernel_operation/cagra_bam_pq_blocksearch.cuh"
#include "../graph_kernel_operation/cagra_bam_pq_blocksearch_kselect.cuh"
#include "../graph_kernel_operation/cagra_bam_pq_top1blocksearch.cuh"
#include "../graph_kernel_operation/cagra_in_memory.cuh"
#include "../graph_kernel_operation/kernel_aggregate_forward_edges.cuh"
#include "../graph_kernel_operation/kernel_detourable_route_count.cuh"
#include "../graph_kernel_operation/kernel_detourable_route_sort.cuh"
#include "../graph_kernel_operation/kernel_global_edge_sort.cuh"
#include "../graph_kernel_operation/kernel_local_graph_construction.cuh"
#include "../graph_kernel_operation/kernel_local_graph_mergence_nsw.cuh"
#include "../graph_kernel_operation/kernel_local_neighbors_sort_nsw.cuh"
#include "../graph_kernel_operation/kernel_reversed_detourable_route_build.cuh"
#include "../graph_kernel_operation/kernel_reversed_detourable_route_sort.cuh"
#include "../graph_kernel_operation/kernel_search_nsw.cuh"
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "auto_tune_bloom.h"

#include "../graph_kernel_operation/bamWriteSSD.cuh"
#include "../graph_kernel_operation/settings.cuh"
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "./pq.cuh"
#include "./kernel_populate_chunk_distances.cuh"
#include "../../common.h"
#include "../../RVQ/RVQ.cuh"
using namespace std;

class NSWGraphOperations {
public:

	static void LocalGraphConstructionBruteForce(float* h_data, int offset_shift, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors,
											int num_of_batches, int num_of_points_one_batch, float* &d_data, KernelPair<float, int>* &d_neighbors,
											KernelPair<float, int>* &d_neighbors_backup);
	
	static void LocalGraphMergenceCoorperativeGroup(float*& d_data, int*& h_graph, int*& d_graph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_initial_neighbors, int num_of_batches, 
														int num_of_points_one_batch, KernelPair<float, int>*& d_neighbors, KernelPair<float, int>*& d_neighbors_backup,
														int num_of_final_neighbors, int num_of_candidates, pair<float, int>* first_subgraph, float* h_distance, float*& d_distance);
	
	static void SubgraphMergence(string sub_data_path, string sub_graph_path, string sub_distance_path, int num_of_subgraph, int total_num_of_points, int dim_of_point, int offset_shift, 
								 int num_of_final_neighbors, int num_of_initial_neighbors, int num_of_candidates, int* pre_fix_of_subgraph_size);
	
	static void Search(array_t<char> *ssd_data, float* h_data, float* d_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, 
						int offset_shift, int num_of_topk, int num_of_candidates, int num_of_explored_points, int* d_enter_cluster, GPUIndex* d_rvq_index, Timer* &graphSearch, int search_width,
                        int num_of_warmup_vectors, float *d_warmup_vectors); 

};

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_SYNC_CHECK()                                                      \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t res = cudaGetLastError();                                  \
        if (res != cudaSuccess) {                                              \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(res));                                  \
            exit(2);                                                           \
        }                                                                      \
    }

cudaError_t error_check(cudaError_t error_code, int line) {
    if (error_code != cudaSuccess) {
        printf(
            "line: %d, error_code: %d, error_name: %s, error_description: %s\n",
            line, error_code, cudaGetErrorName(error_code),
            cudaGetErrorString(error_code));
    }
    return error_code;
}

__global__ void ConvertNeighborstoGraph(int *d_graph,
                                        KernelPair<float, int> *d_neighbors,
                                        int total_num_of_points,
                                        int offset_shift, int num_of_iterations,
                                        float *d_distance) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = (1 << offset_shift) / 2;

    for (int i = 0; i < num_of_iterations; i++) {
        int crt_point_id = i * base + b_id;
        KernelPair<float, int> *crt_neighbors =
            d_neighbors + (size_t(crt_point_id) << offset_shift);
        int *crt_graph = d_graph + size_t(crt_point_id) * size_t(num_of_final_neighbors);
        float *crt_distance =
            d_distance + size_t(crt_point_id) * size_t(num_of_final_neighbors);
        if (crt_point_id < total_num_of_points) {
            for (int j = 0;
                 j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp;
                 j++) {
                int unroll_t_id = t_id + size_of_warp * j;

                if (unroll_t_id < num_of_final_neighbors) {
                    crt_graph[unroll_t_id] = crt_neighbors[unroll_t_id].second;
                    crt_distance[unroll_t_id] =
                        crt_neighbors[unroll_t_id].first;
                }
            }
        }
    }
}

__global__ void LoadFirstSubgraph(pair<float, int> *first_subgraph,
                                  KernelPair<float, int> *d_first_subgraph,
                                  int num_of_copied_edges) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int global_t_id = b_id * blockDim.x + t_id;

    if (global_t_id < num_of_copied_edges) {
        d_first_subgraph[global_t_id].first = first_subgraph[global_t_id].first;
        d_first_subgraph[global_t_id].second =
            first_subgraph[global_t_id].second;
    }
}

void NSWGraphOperations::LocalGraphConstructionBruteForce(
    float *h_data, int offset_shift, int total_num_of_points, int dim_of_point,
    int num_of_initial_neighbors, int num_of_batches,
    int num_of_points_one_batch, float *&d_data,
    KernelPair<float, int> *&d_neighbors,
    KernelPair<float, int> *&d_neighbors_backup) {
    
    int num_of_batches_tmp = 800;
    KernelPair<float, int> *d_distance_matrix;
    cudaMalloc(&d_distance_matrix, size_t(num_of_batches_tmp) * size_t(num_of_points_one_batch) *
                           size_t(num_of_points_one_batch) * sizeof(KernelPair<float, int>));
    error_check(cudaGetLastError(), __LINE__);

    for (int i = 0; i * num_of_batches_tmp < num_of_batches; i++) {
        // KernelPair<float, int> *d_distance_matrix;
        // error_check(cudaMalloc(&d_distance_matrix,
        //                        num_of_batches_tmp * num_of_points_one_batch *
        //                            num_of_points_one_batch *
        //                            sizeof(KernelPair<float, int>)),
        //             __LINE__);
        // printf("%d : %d\n", num_of_batches, i * num_of_batches_tmp);
        DistanceMatrixComputation<<<num_of_batches_tmp, 32>>>(
            d_data, total_num_of_points, num_of_points_one_batch,
            d_distance_matrix, i);
        error_check(cudaGetLastError(), __LINE__);

        SortNeighborsonLocalGraph<<<num_of_points_one_batch, 32,
                                    2 * num_of_initial_neighbors *
                                        sizeof(KernelPair<float, int>)>>>(
            d_neighbors, d_neighbors_backup, total_num_of_points, d_data,
            num_of_points_one_batch, num_of_initial_neighbors, offset_shift,
            d_distance_matrix, i, num_of_batches_tmp);
        error_check(cudaGetLastError(), __LINE__);
    }
    cudaFree(d_distance_matrix);
}
__global__ void initializeReversedGraph(int *d_reversed_graph,
                                        int *d_num_of_reversed_detourable_route,
                                        int num_of_initial_neighbors,
                                        int total_num_of_points,
                                        int num_of_iterations) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    for (int i = 0; i < num_of_iterations; i++) {
        int crt_point_id = i * base + b_id;
        int *crt_graph =
            d_reversed_graph + crt_point_id * num_of_initial_neighbors;
        int *crt_num_of_reversed_detourable_route =
            d_num_of_reversed_detourable_route +
            crt_point_id * num_of_initial_neighbors;
        if (crt_point_id < total_num_of_points) {
            for (int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) /
                                    size_of_warp;
                 j++) {
                int unroll_t_id = t_id + size_of_warp * j;

                if (unroll_t_id < num_of_initial_neighbors) {
                    crt_graph[unroll_t_id] = total_num_of_points;
                    crt_num_of_reversed_detourable_route[unroll_t_id] = MAX;
                }
            }
        }
    }
}

void NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(
    float *&d_data, int *&h_graph, int*& d_graph, int total_num_of_points, int dim_of_point,
    int offset_shift, int num_of_initial_neighbors, int num_of_batches,
    int num_of_points_one_batch, KernelPair<float, int> *&d_neighbors,
    KernelPair<float, int> *&d_neighbors_backup, int num_of_final_neighbors,
    int num_of_candidates, pair<float, int> *first_subgraph,
    float *h_distance, float*& d_distance) {


    Edge *d_edge_all_blocks;
    int *d_flag_all_blocks;
    int num_of_forward_edges;
    num_of_forward_edges = pow(2.0, ceil(log(num_of_points_one_batch) / log(2))) * num_of_initial_neighbors;
    cudaMalloc(&d_edge_all_blocks, num_of_forward_edges * sizeof(Edge));

    cudaMalloc(&d_flag_all_blocks, (num_of_forward_edges + 1) * sizeof(int));
    error_check(cudaGetLastError(), __LINE__);

    pair<float, int> *d_first_subgraph;
    cudaMalloc(&d_first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(pair<float, int>));
    cudaMemcpy(d_first_subgraph, first_subgraph,
               num_of_points_one_batch * num_of_final_neighbors *
                   sizeof(pair<float, int>),
               cudaMemcpyHostToDevice);
    error_check(cudaGetLastError(), __LINE__);

    LoadFirstSubgraph<<<num_of_points_one_batch, num_of_final_neighbors>>>(
        d_first_subgraph, d_neighbors,
        num_of_points_one_batch * num_of_final_neighbors);
    error_check(cudaGetLastError(), __LINE__);
    KernelPair<float, int>* h_neighbors_backup = new KernelPair<float, int>[size_t(total_num_of_points) << offset_shift];
    cudaMemcpy(h_neighbors_backup, d_neighbors, (size_t(total_num_of_points) << offset_shift) * sizeof(KernelPair<float, int>), cudaMemcpyDeviceToHost);
    cudaMalloc(&d_neighbors_backup, (size_t(num_of_points_one_batch) << offset_shift) * sizeof(KernelPair<float, int>));
    error_check(cudaGetLastError(), __LINE__);
    
    for (int i = 1; i < num_of_batches; i++) {
        // printf("%d : %d\n", num_of_batches, i);
        int copy_points = num_of_points_one_batch;
        if(i == num_of_batches - 1){
            copy_points = total_num_of_points - num_of_points_one_batch * i;
        }
        cudaMemcpy(d_neighbors_backup, h_neighbors_backup + (size_t(i * num_of_points_one_batch) << offset_shift), 
                  (size_t(copy_points) << offset_shift) * sizeof(KernelPair<float, int>), cudaMemcpyHostToDevice);
        LocalGraphMergence<<<num_of_points_one_batch, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>
                            (d_neighbors, d_neighbors_backup, total_num_of_points, d_data,
                            d_edge_all_blocks, i, num_of_points_one_batch,
                            num_of_final_neighbors + num_of_candidates, num_of_final_neighbors,
                            num_of_candidates, num_of_initial_neighbors, offset_shift);

        dim3 grid_of_kernel_edge_sort(num_of_forward_edges / 128, 1, 1);
        dim3 block_of_kernel_edge_sort(128, 1, 1);

        int num_of_valid_edges =
            num_of_points_one_batch * num_of_initial_neighbors;
        if (i == num_of_batches - 1) {
            num_of_valid_edges =
                (total_num_of_points -
                 (num_of_batches - 1) * num_of_points_one_batch) *
                num_of_initial_neighbors;
        }

        void *kernel_args[] = {
            (void *)&d_neighbors,        (void *)&d_edge_all_blocks,
            (void *)&d_flag_all_blocks,  (void *)&num_of_forward_edges,
            (void *)&num_of_valid_edges, (void *)&total_num_of_points};

        // sort for edges
        cudaLaunchCooperativeKernel((void *)GlobalEdgesSort,
                                    grid_of_kernel_edge_sort,
                                    block_of_kernel_edge_sort, kernel_args, 0);

        int num_of_types_valid_edges = 0;
        cudaMemcpy(&num_of_types_valid_edges,
                   d_flag_all_blocks + num_of_forward_edges, sizeof(int),
                   cudaMemcpyDeviceToHost);

        AggragateForwardEdges<<<num_of_types_valid_edges, 32, 2 * num_of_final_neighbors * sizeof(KernelPair<float, int>)>>>
                            (d_neighbors, d_edge_all_blocks, d_flag_all_blocks,
                            total_num_of_points, num_of_final_neighbors, offset_shift);
    }
    cudaDeviceSynchronize();
    delete[] h_neighbors_backup;
    cudaFree(d_first_subgraph);
    cudaFree(d_edge_all_blocks);
    cudaFree(d_flag_all_blocks);
    cudaFree(d_data);
    cudaFree(d_neighbors_backup);
    size_t free_mem, total_mem; 
    // cudaMemGetInfo(&free_mem, &total_mem); 
    // printf("Free GPU memory: %zu bytes, Total GPU memory: %zu bytes\n", free_mem, total_mem); 
    // printf("Attempting to allocate: %zu bytes\n", sizeof(int) * (size_t(total_num_of_points) * size_t(num_of_initial_neighbors)) 
    //         + sizeof(float) * (size_t(total_num_of_points) * size_t(num_of_initial_neighbors)));

    int num_of_blocks = 10000;
    int num_of_iterations = (total_num_of_points + num_of_blocks - 1) / num_of_blocks;
    
    cudaMalloc(&d_graph, sizeof(int) * (size_t(total_num_of_points) * size_t(num_of_initial_neighbors)));

    cudaMalloc(&d_distance, sizeof(float) * (size_t(total_num_of_points) * size_t(num_of_initial_neighbors)));
    error_check(cudaGetLastError(), __LINE__);
    ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph, d_neighbors, total_num_of_points, offset_shift,
                                                    num_of_iterations, d_distance);
    cudaDeviceSynchronize();
    // cudaMemcpy(h_distance, d_distance, sizeof(float) * (total_num_of_points
    // * num_of_initial_neighbors), cudaMemcpyDeviceToHost);
    h_graph = new int[size_t(total_num_of_points) * size_t(num_of_initial_neighbors)];
    cudaMemcpy(h_graph, d_graph, sizeof(int) * size_t(total_num_of_points) * size_t(num_of_initial_neighbors), cudaMemcpyDeviceToHost);
    cudaFree(d_neighbors);
    
    // // Graph optimization
    // printf("Graph optimization\n");
    // int* d_num_of_detourable_route;
    // cudaMalloc(&d_num_of_detourable_route, sizeof(int) * (total_num_of_points
    // << offset_shift)); cudaMemset(d_num_of_detourable_route, 0, sizeof(int) *
    // (total_num_of_points << offset_shift));
    // countDetourableRoute<<<num_of_blocks, 32>>>(d_graph,
    // d_num_of_detourable_route, offset_shift, num_of_iterations,
    // total_num_of_points);
    // //reordering and pruning
    // int* d_reversed_graph;
    // cudaMalloc(&d_reversed_graph, sizeof(int) * total_num_of_points *
    // num_of_initial_neighbors); int* d_num_of_reversed_neighbors;
    // cudaMalloc(&d_num_of_reversed_neighbors, sizeof(int) *
    // total_num_of_points); cudaMemset(d_num_of_reversed_neighbors, 0,
    // sizeof(int) * total_num_of_points); int*
    // d_num_of_reversed_detourable_route;
    // cudaMalloc(&d_num_of_reversed_detourable_route, sizeof(int) *
    // total_num_of_points * num_of_initial_neighbors);
    // initializeReversedGraph<<<num_of_blocks, 32>>>(d_reversed_graph,
    // d_num_of_reversed_detourable_route, num_of_initial_neighbors,
    // total_num_of_points, num_of_iterations); constexpr int WARP_SIZE = 32;
    // constexpr int NumWarpQ = 32;
    // constexpr int NumThreadQ = 2;

    // sortDetourableRoute<int, float, WARP_SIZE, NumWarpQ,
    // NumThreadQ><<<num_of_blocks, 32>>>(d_graph, d_num_of_detourable_route,
    // offset_shift, num_of_initial_neighbors, num_of_iterations,
    // total_num_of_points);
    // //build and merge reversed graph
    // buildReversedDetourableRoute<<<num_of_blocks, 32>>>(d_graph,
    // d_num_of_detourable_route, d_reversed_graph, d_num_of_reversed_neighbors,
    // d_num_of_reversed_detourable_route,
    //                                          			offset_shift,
    //                                          num_of_initial_neighbors,
    //                                          num_of_iterations,
    //                                          total_num_of_points);

    // sortReversedDetourableRoute<int, float, WARP_SIZE, NumWarpQ,
    // NumThreadQ><<<num_of_blocks, 32>>>(d_graph, d_reversed_graph,
    // d_num_of_reversed_neighbors, d_num_of_reversed_detourable_route,
    //                                         														offset_shift,
    //                                         num_of_initial_neighbors,
    //                                         num_of_iterations,
    //                                         total_num_of_points);
    // cudaMemcpy(h_graph, d_reversed_graph, sizeof(int) * total_num_of_points *
    // num_of_initial_neighbors, cudaMemcpyDeviceToHost); cudaFree(d_graph);
    // cudaFree(d_num_of_detourable_route);
    // cudaFree(d_reversed_graph);
    // cudaFree(d_num_of_reversed_neighbors);
    // cudaFree(d_num_of_reversed_detourable_route);
}

__global__ void zeroCount(int *d_enter_cluster, int *d_rvq_index_sizes,
                          int *d_num_of_zero_query, int *d_max_cluster) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cluster_id = d_enter_cluster[bid];
    int cluster_size = d_rvq_index_sizes[cluster_id];
    if (tid == 0) {
        // printf("%d ",cluster_size);
        if (cluster_size == 0) {
            atomicAdd(&d_num_of_zero_query[0], 1);
        }
        atomicMax(&d_max_cluster[0], cluster_size);
    }
}

void readGraph(string path, int *graph, int num_of_subgraph,
               int num_of_candidates, int num_of_neighbors,
               int total_num_of_points, int *pre_fix_of_subgraph_size) {
    for (int i = 0; i < num_of_subgraph; i++) {
        ostringstream graph_filename;
        graph_filename << "finalGraph" << std::setw(4) << std::setfill('0') << i
                       << "_" << 32 << "_" << 16 << "_"
                       << to_string(total_num_of_points / 1000000) << "M"
                       << ".nsw";
        string main_graph_path = path + graph_filename.str();
        ifstream in_descriptor(main_graph_path, std::ios::binary);
        in_descriptor.read(
            (char *)(graph + pre_fix_of_subgraph_size[i] * num_of_neighbors),
            sizeof(int) *
                (pre_fix_of_subgraph_size[i + 1] -
                 pre_fix_of_subgraph_size[i]) *
                num_of_neighbors);
        in_descriptor.close();
    }
}

void readCagraGraph(string path, int n, int dim, int *graph) {
    ifstream in_descriptor(path, std::ios::binary);

    if (!in_descriptor.is_open()) {
        exit(1);
    }

    in_descriptor.seekg(790, std::ios::beg);
    in_descriptor.read((char *)(graph), (n << dim) * sizeof(int));
    // int data_size = 800;
    // char neighbors[800];
    // in_descriptor.read((char*)(neighbors), data_size * sizeof(char));
    // for(int i = 0; i < data_size; i++){
    //     cout<<neighbors[i];
    // }
    // cout<<endl;
    // in_descriptor.close();
}

__global__ void queryCopy(float *d_query, int dim, int copy_id) {
    int b_id = blockIdx.x;
    int t_id = threadIdx.x;
    int size_of_block = blockDim.x;
    if (b_id == copy_id)
        return;
    float *copy_from = d_query + dim * copy_id;
    float *copy_to = d_query + dim * b_id;
    for (int i = 0; i < (dim + size_of_block - 1) / size_of_block; i++) {
        int unrollid = t_id + i * size_of_block;
        if (unrollid < dim) {
            copy_to[unrollid] = copy_from[unrollid];
        }
    }
}

__global__ void read_graph_kernel(array_d_t<char> *d_array, size_t num_of_reads) {
    size_t t_id = threadIdx.x;
    size_t idx = blockIdx.x;
    size_t num_of_blocks = gridDim.x;
    for(int i = 0; i < (num_of_reads + num_of_blocks - 1) / num_of_blocks; i++){
        size_t block_id = idx + i * num_of_blocks;
        if(block_id < num_of_reads){
            size_t offset = 4096 * block_id;
            char value = (*d_array)[offset + t_id];
        }
    }
    
}

__global__ void read_data_kernel(array_d_t<char> *d_array, size_t point_id, size_t dim, size_t degree, size_t numElementsPerBlock) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t page_size = 4096;
    if (idx < dim + degree) {
        size_t offset = (point_id / numElementsPerBlock) * page_size / sizeof(int) + (point_id % numElementsPerBlock) * (dim + degree + 1);
        array_d_t<float>* value = reinterpret_cast<array_d_t<float>*>(d_array);
        array_d_t<int>* neighbor = reinterpret_cast<array_d_t<int>*>(d_array);
        if(idx < dim){
            // array_d_t<float>* value = reinterpret_cast<array_d_t<float>*>(d_array);
            printf("Data idx = %lu, value = %f\n", idx, (*value)[idx + offset]);
        }
        else if(idx < dim + degree){
            // array_d_t<int>* neighbor = reinterpret_cast<array_d_t<int>*>(d_array);
            size_t num_of_neighbor = (*neighbor)[dim];
            if(idx < dim + num_of_neighbor){
                printf("Neighbor idx = %lu, value = %d\n", idx - dim, (*neighbor)[1 + idx + offset]);
            }
        }
    }
}

// template<typename T>
// __global__ void compare_arrays(T* d_array1, array_d_t<T>* d_array2, int size, int* diff_indices, int* diff_count) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < size) {
//         T value1 = d_array1[idx];
//         T value2 = (*d_array2)[idx];

//         if (value1 != value2) {
//             int diff_idx = atomicAdd(diff_count, 1);
//             diff_indices[diff_idx] = idx;
//         }
//     }
// }

__global__ void prefetch(array_d_t<char> *ssd_data, int* id_array, size_t degree_of_graph, size_t hops, size_t dim_of_point, size_t page_size, 
                        size_t num_elements_per_block, int* d_starling_index){
    int t_id = threadIdx.x;
    size_t size_of_block = size_t(blockDim.x);
    id_array[t_id] = t_id;
    // printf("id_array[%d]: %d\n", t_id, id_array[t_id]);
    // size_t offset = (size_t(d_starling_index[t_id]) / num_elements_per_block) * page_size / sizeof(int) + 
    //                 (size_t(d_starling_index[t_id]) % num_elements_per_block) * (dim_of_point + degree_of_graph + 1);
    array_d_t<float>* vector_data = reinterpret_cast<array_d_t<float>*>(ssd_data);
    array_d_t<int>* neighbors_data = reinterpret_cast<array_d_t<int>*>(ssd_data);
    // float tmp_vector_data = (*vector_data)[offset];
    // printf("vector: %f, data: %d\n", vector_data, neighbors_data);
    size_t num_of_total_fetch_points = 0;
    __syncthreads();
    for(size_t i = 1; i < hops; i++){
       int* start_of_fetch = id_array + num_of_total_fetch_points;
       size_t num_of_fetch_points = degree_of_graph * pow(degree_of_graph, i - 1);
    //    printf("num_of_total_fetch_points: %lu,num_of_fetch_points: %lu\n", num_of_total_fetch_points, num_of_fetch_points);
       num_of_total_fetch_points += num_of_fetch_points;
       for(size_t l = 0; l < (num_of_fetch_points + size_of_block - 1) / size_of_block; l++){
            size_t fetch_loc = size_t(t_id) + size_of_block * l;
            if(fetch_loc < num_of_fetch_points){
                // printf("%d : %d\n",start_of_fetch[fetch_loc], d_starling_index[start_of_fetch[fetch_loc]]);
                int target_point_id = d_starling_index[start_of_fetch[fetch_loc]];
                size_t offset = (size_t(target_point_id) / num_elements_per_block) * page_size / sizeof(int) + 
                         (size_t(target_point_id) % num_elements_per_block) * (dim_of_point + degree_of_graph + 1) + dim_of_point;
                int num_of_neighbors = (*neighbors_data)[offset];
                // printf("idx: %d num_of_neighbors: %d\n", target_point_id, num_of_neighbors);
                int* add_of_array = id_array + num_of_total_fetch_points + fetch_loc * degree_of_graph;
                for(size_t k = 0; k < num_of_neighbors; k++){
                    int tmp_neighbor_data = (*neighbors_data)[offset + 1 + k];
                    add_of_array[k] = tmp_neighbor_data;
                }
            }
       }
       __syncthreads();
    }
}
__global__ void get_distance(size_t query_id, size_t target, size_t degree_of_point, size_t dim, size_t page_size, size_t num_elements_per_block, 
                             float* query, array_d_t<char> *ssd_data){
    int t_id = threadIdx.x;
    array_d_t<float>* vector_data = reinterpret_cast<array_d_t<float>*>(ssd_data);
    size_t offset = (target / num_elements_per_block) * page_size / sizeof(int) + (target % num_elements_per_block) * (dim + degree_of_point + 1);
    if(t_id == 0){
        float dist = 0;
        for(size_t i = 0; i < dim; i++){
            // if(i == 0){
            //     printf("First element %f\n", (*ssd_data)[target * dim + i]);
            // }
            float tmp_dist = query[query_id * dim + i] - (*vector_data)[offset + i];
            dist += tmp_dist * tmp_dist;
        }
        printf("The distance from query %lu to point %lu is %f, offset %lu\n", query_id, target, dist, target * dim);
    }
    
}
__global__ void index2starling(int** d_rvq_indices, int* d_rvq_indices_size, int *d_starling_index, int num_of_cluster){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int num_of_blocks = gridDim.x;
    int num_of_threads = blockDim.x;
    for(int i = 0; i < (num_of_cluster + num_of_blocks - 1) / num_of_blocks; i++){
        int cluster_id = b_id + i * num_of_blocks;
        if(cluster_id < num_of_cluster){
            int cluster_size = d_rvq_indices_size[cluster_id];
            int* cluster = d_rvq_indices[cluster_id];
            for(int l = 0; l < (cluster_size + num_of_threads - 1) / num_of_threads; l++){
                int idx = t_id + l * num_of_threads;
                if(idx < cluster_size){
                    cluster[idx] = d_starling_index[cluster[idx]];
                }
            }
        }
    }
}

void load_vamana_graph(std::string data_path, std::string graph_path, std::string tags_path, float *&cache_data, int *&cache_graph, int *&cache_tags, 
                       int &num_of_cache_points, int &degree_of_cache_point){
    int dim_of_points;
    std::ifstream data_in(data_path, std::ios::binary);
    data_in.read((char*)&num_of_cache_points, sizeof(int));
    data_in.read((char*)&dim_of_points, sizeof(int));
    cache_data = new float[num_of_cache_points * dim_of_points];
    data_in.read((char*)cache_data, size_t(num_of_cache_points) * size_t(dim_of_points) * sizeof(float));
    data_in.close();
    std::ifstream tags_in(tags_path, std::ios::binary);
    tags_in.seekg(2 * sizeof(int), std::ios::beg);
    cache_tags = new int[num_of_cache_points];
    tags_in.read((char*)cache_tags, size_t(num_of_cache_points) * sizeof(int));
    tags_in.close();

    std::ifstream graph_in(graph_path, std::ios::binary);
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(unsigned);
    graph_in.seekg(sizeof(size_t), std::ios::beg);
    graph_in.read((char*)&degree_of_cache_point, sizeof(int));
    graph_in.seekg(header_size, std::ios::beg);
    degree_of_cache_point = pow(2, ceil(log(degree_of_cache_point) / log(2)));
    cache_graph = new int[size_t(degree_of_cache_point) * size_t(num_of_cache_points)];
    for(size_t i = 0; i < num_of_cache_points; i++){
        int k;
        int *crt_cache_graph = cache_graph + i * size_t(degree_of_cache_point);
        graph_in.read((char*)&k, sizeof(int));
        graph_in.read((char*)crt_cache_graph, size_t(k) * sizeof(int));
        for(int l = k; l < degree_of_cache_point; l++){
            crt_cache_graph[l] = num_of_cache_points;
        }
    }
    graph_in.close();

}
void NSWGraphOperations::Search(array_t<char> *ssd_data, float *d_data, float *d_query, int *h_graph,
                                int *h_result, int num_of_query_points,
                                int total_num_of_points, int dim_of_point,
                                int offset_shift, int num_of_topk,
                                int num_of_candidates,
                                int num_of_explored_points,
                                int *d_enter_cluster, GPUIndex *d_rvq_index,
                                Timer *&graphSearch, int search_width, int num_of_warmup_vectors, float *d_warmup_vectors) { //ssd_data->d_array_ptr
    graphSearch[1].Start();
    int *d_result;
    cudaMalloc(&d_result, sizeof(int) * (num_of_topk * num_of_query_points));
    graphSearch[1].Stop();
    CUDA_SYNC_CHECK();
    float epsilon = 1;

    size_t page_size = 4096;
    size_t element_size = dim_of_point * sizeof(float) + ((size_t(1) << offset_shift) + 1) * sizeof(int);
    size_t num_elements_per_block = page_size / element_size;
    int length_of_block = pow(2.0, ceil(log(int(num_elements_per_block)) / log(2)));
    size_t num_of_blocks = ceil(size_t(total_num_of_points) / num_elements_per_block);
    printf("num_elements_per_block:%lu, length_of_block:%d\n",num_elements_per_block ,length_of_block);
    int *d_num_of_block_accesses;
    cudaMalloc(&d_num_of_block_accesses, size_t(num_of_blocks) * sizeof(int));
    cudaMemset(d_num_of_block_accesses, 0, size_t(num_of_blocks) * sizeof(int));
    int *h_num_of_block_accesses = new int[num_of_blocks];

    std::ifstream in("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/starlingIndex.bin", std::ios::binary);
    int *starling_index = new int[total_num_of_points];
    in.read((char*)starling_index, size_t(total_num_of_points) * sizeof(int));
    in.close();

    int *d_starling_index;
    cudaMalloc(&d_starling_index, size_t(total_num_of_points) * sizeof(int));
    cudaMemcpy(d_starling_index, starling_index, size_t(total_num_of_points) * sizeof(int), cudaMemcpyHostToDevice);

    // index2starling<<<10000, 32>>>(d_rvq_index->indices, d_rvq_index->sizes, d_starling_index, 1000 * 100);
    // cudaFree(d_starling_index);

    std::ifstream resver_in("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/starlingResverIndex.bin", std::ios::binary);
    int* starling_resver_index = new int[total_num_of_points];
    resver_in.read((char*)starling_resver_index, size_t(total_num_of_points) * sizeof(int));
    resver_in.close();

    int *d_starling_resver_index;
    // cudaMalloc(&d_starling_resver_index, size_t(total_num_of_points) * sizeof(int));
    // cudaMemcpy(d_starling_resver_index, starling_resver_index, size_t(total_num_of_points) * sizeof(int), cudaMemcpyHostToDevice);
    
    //IO test
    // Timer IO_time;
    // IO_time.Start();
    // size_t num_of_reads = 2097152;//6815744;
    // read_graph_kernel<<<10000, 32>>>(ssd_data->d_array_ptr, num_of_reads);
    // CUDA_SYNC_CHECK();
    // IO_time.Stop();
    // std::cout<<"IO time: "<<IO_time.DurationInMilliseconds()<<" ms, Number of IOs: " << num_of_reads << ", " 
    // << 1000 * IO_time.DurationInMilliseconds() / num_of_reads <<" us per IO, IOPS: " << num_of_reads / (IO_time.DurationInMilliseconds() / 1000)  << std::endl;
    // ssd_data->print_reset_stats();

    //prefetch
    // Timer prefetch_time;
    // prefetch_time.Start();
    // size_t degree = size_t(1) << offset_shift;
    // size_t hops = 5;
    // size_t num_of_prefetch_points = degree * (pow(degree, hops) - size_t(1)) / (degree - 1);
    // printf("degree: %lu, num_of_prefetch_points: %lu\n", degree, num_of_prefetch_points);
    // int* id_array;
    // cudaMalloc(&id_array, num_of_prefetch_points * sizeof(int));
    // cudaMemset(id_array, 0, num_of_prefetch_points * sizeof(int));
    // prefetch<<<1,degree>>>(ssd_data->d_array_ptr, id_array, degree, hops, size_t(dim_of_point), page_size, num_elements_per_block, d_starling_index);
    // cudaFree(id_array);
    // CUDA_SYNC_CHECK();

    // prefetch_time.Stop();
    // std::cout<<"Prefetch time: "<<prefetch_time.DurationInMilliseconds()<<" ms"<<std::endl;
    // ssd_data->print_reset_stats();


    PQ* pq_index = new PQ();
    pq_index->load_from_separate_paths("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_pq_pivots.bin",
                                        "/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_pq_compressed.bin");
    float *d_centroid;
    cudaMalloc(&d_centroid, pq_index->ndims * sizeof(float));
    cudaMemcpy(d_centroid, pq_index->centroid, pq_index->ndims * sizeof(float), cudaMemcpyHostToDevice);
    uint32_t *d_chunk_offsets; 
    cudaMalloc(&d_chunk_offsets, (pq_index->num_of_chunks + 1) * sizeof(uint32_t));
    cudaMemcpy(d_chunk_offsets, pq_index->chunk_offsets, (pq_index->num_of_chunks + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint8_t *d_pq_data; 
    cudaMalloc(&d_pq_data, uint64_t(total_num_of_points) * pq_index->num_of_chunks * sizeof(uint8_t));
    cudaMemcpy(d_pq_data, pq_index->data, uint64_t(total_num_of_points) * pq_index->num_of_chunks * sizeof(uint8_t), cudaMemcpyHostToDevice);
    float *d_pq_tables;
    cudaMalloc(&d_pq_tables, NUM_PQ_CENTROIDS * pq_index->ndims * sizeof(float));
    cudaMemcpy(d_pq_tables, pq_index->tables_tr, NUM_PQ_CENTROIDS * pq_index->ndims * sizeof(float), cudaMemcpyHostToDevice);

    //warmup
    Timer warmup_time;
    warmup_time.Start();
    int warmup_batch_size = 10000;
    int num_of_warmup_explored_points = 128;
    int num_of_warmup_candidates = pow(2, ceil(log(num_of_warmup_explored_points) / log(2)));
    int num_of_warmup_query;
    int dim_of_warmup_query;
    std::ifstream warmup_in("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/_sample_data.bin", std::ios::binary);
    warmup_in.read((char*)&num_of_warmup_query, sizeof(int));
    warmup_in.read((char*)&dim_of_warmup_query, sizeof(int));
    num_of_warmup_query = min(150000, num_of_warmup_query);
    float *h_warmup_query = new float[num_of_warmup_query * dim_of_warmup_query];
    warmup_in.read((char*)h_warmup_query, num_of_warmup_query * dim_of_warmup_query * sizeof(float));
    warmup_in.close();
    num_of_warmup_query = num_of_warmup_vectors;
    std::cout<<"Start warmup......Number of warmup query: "<< num_of_warmup_query <<std::endl;

    for(int i = 0; i < (num_of_warmup_query + warmup_batch_size - 1) / warmup_batch_size; i++){
        // std::cout<<"Warmup......Iteration: "<< i <<std::endl;
        int num_of_batch_query = min(warmup_batch_size, num_of_warmup_query - warmup_batch_size * i);
        float *d_warmup_query;
        // cudaMalloc(&d_warmup_query, num_of_batch_query * dim_of_warmup_query * sizeof(float));
        // cudaMemcpy(d_warmup_query, h_warmup_query + i * warmup_batch_size * dim_of_warmup_query, num_of_batch_query * dim_of_warmup_query * sizeof(float), cudaMemcpyHostToDevice);
        d_warmup_query = d_warmup_vectors + i * warmup_batch_size * dim_of_warmup_query;
        float *d_dist_vec;
        cudaMalloc(&d_dist_vec, size_t(num_of_batch_query) * NUM_PQ_CENTROIDS * pq_index->num_of_chunks * sizeof(float));
        cudaMemset(d_dist_vec, 0, size_t(num_of_batch_query) * NUM_PQ_CENTROIDS * pq_index->num_of_chunks * sizeof(float));
        populate_chunk_distances<<<num_of_batch_query, 32>>>(size_t(num_of_batch_query), size_t(dim_of_warmup_query), pq_index->num_of_chunks, d_warmup_query, 
                                                            d_dist_vec, d_centroid, d_chunk_offsets, d_pq_data, d_pq_tables);
        CUDA_SYNC_CHECK();
        size_t hash_size = 100 * 1024 / sizeof(uint32_t);
        uint32_t* d_hash_table;
        cudaMalloc(&d_hash_table, hash_size * size_t(num_of_batch_query) * sizeof(uint32_t));
        cudaMemset(d_hash_table, 0, hash_size * size_t(num_of_batch_query) * sizeof(uint32_t));
        cagra_bam_pq_blocksearch_warmup<<<num_of_batch_query, 32, ((4 << offset_shift) * length_of_block + num_of_warmup_candidates) * sizeof(KernelPair<float, int>)>>>
        (ssd_data->d_array_ptr, d_warmup_query, total_num_of_points, offset_shift, num_of_warmup_candidates, num_of_warmup_explored_points, 
        4, hash_size, d_hash_table, page_size, num_elements_per_block, element_size / sizeof(int), length_of_block, d_dist_vec, 
        d_pq_data, pq_index->num_of_chunks, d_starling_index);
        // cudaFree(d_warmup_query);
        cudaFree(d_dist_vec);
        cudaFree(d_hash_table);
        CUDA_SYNC_CHECK();
    }
    CUDA_SYNC_CHECK();
    delete[] h_warmup_query;
    warmup_time.Stop();
    ssd_data->print_reset_stats();
    std::cout<<"Warmup time: "<<warmup_time.DurationInMilliseconds()<<" ms"<<std::endl;
    
    int num_of_cache_candidates = 1;
    // num_of_cache_candidates = min(num_of_cache_candidates, num_of_candidates);
    float *h_cache_data;
    int *h_cache_graph;
    int *h_cache_tags;
    int num_of_cache_points;
    int degree_of_cache_point;
    std::string data_path = "/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/MEM_R_48_L_128_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.01_FREQ_RATE0.01/_index.data";
    std::string graph_path = "/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/MEM_R_48_L_128_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.01_FREQ_RATE0.01/_index";
    std::string tags_path = "/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/MEM_R_48_L_128_ALPHA_1.2_MEM_USE_FREQ0_RANDOM_RATE0.01_FREQ_RATE0.01/_index.tags";
    load_vamana_graph(data_path, graph_path, tags_path, h_cache_data, h_cache_graph, h_cache_tags, num_of_cache_points, degree_of_cache_point);
    int cache_offset_shift = ceil(log(degree_of_cache_point) / log(2));
    float *d_cache_data;
    cudaMalloc(&d_cache_data, size_t(num_of_cache_points) * size_t(dim_of_point) * sizeof(float));
    cudaMemcpy(d_cache_data, h_cache_data, size_t(num_of_cache_points) * size_t(dim_of_point) * sizeof(float), cudaMemcpyHostToDevice);
    int *d_cache_graph;
    cudaMalloc(&d_cache_graph, size_t(num_of_cache_points) * size_t(degree_of_cache_point) * sizeof(int));
    cudaMemcpy(d_cache_graph, h_cache_graph, size_t(num_of_cache_points) * size_t(degree_of_cache_point) * sizeof(int), cudaMemcpyHostToDevice);
    int *d_cache_tags;
    cudaMalloc(&d_cache_tags, size_t(num_of_cache_points) * sizeof(int));
    cudaMemcpy(d_cache_tags, h_cache_tags, size_t(num_of_cache_points) * sizeof(int), cudaMemcpyHostToDevice);
    int *d_cache_result;
    cudaMalloc(&d_cache_result, size_t(num_of_cache_candidates) * size_t(num_of_query_points) * sizeof(int));
    // printf("cache points %d, degree: %d, offset: %d\n", num_of_cache_points, degree_of_cache_point, cache_offset_shift);
    int cache_search_width = 1;
    Timer search_cache_graph_time;
    search_cache_graph_time.Start();
    cagra_in_memory<<<num_of_query_points, 64, ((cache_search_width << cache_offset_shift) + num_of_cache_candidates) * sizeof(KernelPair<float, int>)>>>
    (d_cache_data, d_query, d_cache_result, d_cache_graph, num_of_cache_points, cache_offset_shift, num_of_cache_candidates, 
    num_of_cache_candidates, num_of_cache_candidates, cache_search_width, d_cache_tags);
    CUDA_SYNC_CHECK();
    search_cache_graph_time.Stop();
    std::cout<<"Search cache graph time: "<<search_cache_graph_time.DurationInMilliseconds()<<" ms"<<std::endl;

    // get_distance<<<1, 1>>>(0, 83706315, size_t(1) << offset_shift, size_t(dim_of_point), page_size, num_elements_per_block, 
    //                        d_query, ssd_data);

    //Hash
    size_t hash_size = 100 * 1024 / sizeof(uint32_t);
    uint32_t* d_hash_table;
    cudaMalloc(&d_hash_table, hash_size * size_t(num_of_query_points) * sizeof(uint32_t));
    cudaMemset(d_hash_table, 0, hash_size * size_t(num_of_query_points) * sizeof(uint32_t));
    CUDA_SYNC_CHECK();

    Timer populate_chunk_distances_time;
    populate_chunk_distances_time.Start();
    float *d_query_vec;
    cudaMalloc(&d_query_vec, num_of_query_points * dim_of_point * sizeof(float));
    cudaMemcpy(d_query_vec, d_query, num_of_query_points * dim_of_point * sizeof(float), cudaMemcpyDeviceToDevice);
    float *d_dist_vec;
    cudaMalloc(&d_dist_vec, size_t(num_of_query_points) * NUM_PQ_CENTROIDS * pq_index->num_of_chunks * sizeof(float));
    cudaMemset(d_dist_vec, 0, size_t(num_of_query_points) * NUM_PQ_CENTROIDS * pq_index->num_of_chunks * sizeof(float));
    populate_chunk_distances<<<num_of_query_points, 32>>>(size_t(num_of_query_points), size_t(dim_of_point), pq_index->num_of_chunks, d_query_vec, 
                                                          d_dist_vec, d_centroid, d_chunk_offsets, d_pq_data, d_pq_tables);
    CUDA_SYNC_CHECK();
    populate_chunk_distances_time.Stop();
    cudaFree(d_query_vec);
    cout<< "Initialize PQ: " << populate_chunk_distances_time.DurationInMilliseconds() <<" ms" << endl;

    //统计
    unsigned long long* h_time_breakdown;
	unsigned long long* d_time_breakdown;
	int num_of_phases = 4;
	cudaMallocHost(&h_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	cudaMalloc(&d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	cudaMemset(d_time_breakdown, 0, num_of_query_points * num_of_phases * sizeof(unsigned long long));
    size_t* h_IO_count = new size_t[num_of_query_points];
    size_t* IO_count;
    cudaMalloc(&IO_count, num_of_query_points * sizeof(size_t));
    cudaMemset(IO_count, 0, num_of_query_points * sizeof(size_t));
    int final_num_of_candidates = pow(2, ceil(log(num_of_topk) / log(2)));
    int num_of_points_to_sort = pow(2, ceil(log(num_elements_per_block) / log(2)));
    printf("final_num_of_candidates: %d\n", final_num_of_candidates);

    constexpr int WARP_SIZE = 32;
    constexpr int NumWarpQ = 32;
    constexpr int NumThreadQ = 16;
    length_of_block = pow(2, log(num_elements_per_block) / log(2));
    graphSearch[2].Start();
    // size_t query_batch = 5000;
    // size_t iteration = (num_of_query_points + query_batch - 1) / query_batch;
    // for(size_t l = 0; l < iteration; l++){
    //     size_t tmp_query_batch = query_batch;
    //     if(l == iteration - 1) tmp_query_batch = num_of_query_points - query_batch * l;
    //     cagra_bam_pq_blocksearch<<<tmp_query_batch, 32, ((search_width << offset_shift) * length_of_block + num_of_candidates + final_num_of_candidates + num_of_points_to_sort) * sizeof(KernelPair<float, int>)>>>
    //     (ssd_data->d_array_ptr, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, final_num_of_candidates, num_of_topk, num_of_points_to_sort, 
    //     num_of_explored_points, search_width, d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block,
    //     element_size / sizeof(int), length_of_block, d_time_breakdown, IO_count, d_dist_vec, d_pq_data, pq_index->num_of_chunks, d_starling_index, d_starling_resver_index,
    //     d_cache_result, num_of_cache_candidates, d_num_of_block_accesses, query_batch * l);
    // }
    cagra_bam_pq_blocksearch<<<num_of_query_points, 32, ((search_width << offset_shift) * length_of_block + num_of_candidates + final_num_of_candidates + num_of_points_to_sort) * sizeof(KernelPair<float, int>)>>>
    (ssd_data->d_array_ptr, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, final_num_of_candidates, num_of_topk, num_of_points_to_sort, 
    num_of_explored_points, search_width, d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block,
    element_size / sizeof(int), length_of_block, d_time_breakdown, IO_count, d_dist_vec, d_pq_data, pq_index->num_of_chunks, d_starling_index, d_starling_resver_index,
    d_cache_result, num_of_cache_candidates, d_num_of_block_accesses, 0);

    // cagra_bam_pq_top1blocksearch<<<num_of_query_points, 32, ((search_width << offset_shift) + num_of_candidates + final_num_of_candidates + num_of_points_to_sort) * sizeof(KernelPair<float, int>)>>>
    // (ssd_data->d_array_ptr, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, final_num_of_candidates, num_of_topk, num_of_points_to_sort, 
    // num_of_explored_points, search_width, d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block,
    // element_size / sizeof(int), length_of_block, d_time_breakdown, IO_count, d_dist_vec, d_pq_data, pq_index->num_of_chunks, d_starling_index, d_starling_resver_index,
    // d_cache_result, num_of_cache_candidates, d_num_of_block_accesses);

    // cagra_bam_pq_blocksearch_kselect<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<num_of_query_points, 32, (final_num_of_candidates + num_of_points_to_sort) * sizeof(KernelPair<float, int>)>>>
    // (ssd_data->d_array_ptr, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, final_num_of_candidates, num_of_topk, num_of_points_to_sort, 
    // num_of_explored_points, search_width, d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block,
    // element_size / sizeof(int), length_of_block, d_time_breakdown, IO_count, d_dist_vec, d_pq_data, pq_index->num_of_chunks, d_starling_index, d_starling_resver_index,
    // d_cache_result, num_of_cache_candidates, d_num_of_block_accesses);

    // cagra_bam_pq<<<num_of_query_points, 32, ((search_width << offset_shift) + num_of_candidates + final_num_of_candidates + num_of_points_to_sort) * sizeof(KernelPair<float, int>)>>>
    // (ssd_data->d_array_ptr, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, final_num_of_candidates, num_of_topk, num_of_points_to_sort,
    // num_of_explored_points, search_width, d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block, 
    // d_time_breakdown, IO_count, d_dist_vec, d_pq_data, pq_index->num_of_chunks, d_starling_index, d_cache_result, num_of_cache_candidates, d_num_of_block_accesses);
    
    // cagra_bam<int, float, WARP_SIZE, NumWarpQ, NumThreadQ>
    // <<<num_of_query_points, 64,((search_width << offset_shift) + num_of_candidates) * sizeof(KernelPair<float, int>)>>>
    // (ssd_data, d_query, d_result, total_num_of_points, offset_shift, num_of_candidates, num_of_topk, num_of_explored_points, search_width, 
    // d_rvq_index->indices, d_rvq_index->sizes, hash_size, d_hash_table, page_size, num_elements_per_block, d_time_breakdown, IO_count, d_starling_index);
    CUDA_SYNC_CHECK();
    graphSearch[2].Stop();
     cout << "Search Time: " << graphSearch[2].DurationInMilliseconds() <<" ms" << endl;
    graphSearch[2].duration += populate_chunk_distances_time.duration;
    // graphSearch[2].duration += populate_chunk_distances_time.duration + search_cache_graph_time.duration;

    // cudaMemcpy(h_num_of_block_accesses, d_num_of_block_accesses, num_of_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    // string block_accesses_filename = "/home/ErHa/num_of_block_accesses/block_accesses_candidates" + to_string(num_of_candidates) + ".bin";
    // ofstream block_accesses_out_file(block_accesses_filename, ios::binary);
    // block_accesses_out_file.write((char*)(&num_of_blocks), sizeof(int));
    // block_accesses_out_file.write((char*)h_num_of_block_accesses, num_of_blocks * sizeof(int));
    // block_accesses_out_file.close();
    graphSearch[3].Start();
    cudaMemcpy(h_result, d_result, sizeof(int) * num_of_query_points * num_of_topk, cudaMemcpyDeviceToHost);
    graphSearch[3].Stop();

    for(int i = 0; i < num_of_query_points * num_of_topk; i++){
        h_result[i] = starling_resver_index[h_result[i]];
    }
    // for(int i = 0; i < num_of_query_points; i++){
    //     for(int l = 0; l < num_of_topk; l++){
    //         if(i == 0){
    //             printf("%d ", h_result[i * num_of_topk + l]);
    //         }
    //     }
    // }
    // printf("\n");
    for (int i = 0; i < num_of_query_points; i++) {
        int flag = 0;
        for (int l = 0; l < num_of_topk; l++) {
            // printf("%d ", h_result[i * num_of_topk + l]);
            for (int k = l; k < num_of_topk; k++) {
                if (l == k || h_result[i * num_of_topk + l] == total_num_of_points)
                    continue;
                if (h_result[i * num_of_topk + l] == h_result[i * num_of_topk + k]) {
                    printf("%d %d %d\n", i, l, h_result[i * num_of_topk + l]);
                    // for(int j = 0; j < num_of_topk; j++){
                    // 	printf("%d ", h_result[i * num_of_topk + j]);
                    // }
                    // printf("\n");
                    flag = 1;
                    break;
                }
            }
            if (flag == 1)
                break;
        }
        // printf("\n");
    }

    // cudaMemcpy(&h_count, d_count, sizeof(int) * num_of_query_points,
    // cudaMemcpyDeviceToHost); cudaMemcpy(&h_zero_count, d_zero_count,
    // sizeof(int) * num_of_query_points, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_iter, d_iter, sizeof(int) * num_of_query_points,
    // cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_num_of_zero_query,d_num_of_zero_query, sizeof(int),
    // cudaMemcpyDeviceToHost); cudaMemcpy(h_max_cluster,d_max_cluster,
    // sizeof(int), cudaMemcpyDeviceToHost); int h_count_sum = 0; int
    // h_zero_count_sum=0; int h_iter_sum=0; int max_iter = 0; int max_iter_id;
    // for(int i=0; i<num_of_query_points; i++){
    // 	h_count_sum+=h_count[i];
    // 	h_zero_count_sum+=h_zero_count[i];
    // 	h_iter_sum+=h_iter[i];
    // 	if(h_iter[i] > max_iter){
    // 		max_iter = h_iter[i];
    // 		max_iter_id = i;
    // 	}
    // }
    // printf("Max size of query cluster: %d\n", h_max_cluster[0]);
    // printf("Number of query without zero cluster computation: %d\n",
    // h_count_sum); printf("Number of query with zero cluster: %d\n",
    // h_num_of_zero_query[0]); printf("Number of query with zero cluster
    // computation: %d\n", h_zero_count_sum); printf("Number of avg iter: %d\n",
    // h_iter_sum / num_of_query_points); printf("Number of max iter: %d, id:
    // %d\n", max_iter, max_iter_id);

    // cudaFree(d_graph);
    // cudaFree(d_data);
    // cudaFree(d_result);
    // cudaFree(d_count);

    cudaMemcpy(h_time_breakdown, d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long stage_1 = 0;
    unsigned long long stage_2 = 0;
    unsigned long long stage_3 = 0;
    unsigned long long stage_4 = 0;
    // unsigned long long stage_5 = 0;
    // unsigned long long stage_6 = 0;
    // unsigned long long stage_7 = 0;

    for (int i = 0; i < num_of_query_points; i++) {
    	stage_1	+= h_time_breakdown[i * num_of_phases];
    	stage_2	+= h_time_breakdown[i * num_of_phases + 1];
    	stage_3	+= h_time_breakdown[i * num_of_phases + 2];
    	stage_4	+= h_time_breakdown[i * num_of_phases + 3];
    	// stage_5	+= h_time_breakdown[i * num_of_phases + 4];
    	// stage_6	+= h_time_breakdown[i * num_of_phases + 5];
    	// stage_7 += h_time_breakdown[i * num_of_phases + 6];
    }
    unsigned long long sum_of_all_stages = stage_1 + stage_2 + stage_3 + stage_4 /*+ stage_5 + stage_6 + stage_7*/; 
    cout << "stages percentage: " <<
    (double)(stage_1) / sum_of_all_stages << " " << (double)(stage_2) / sum_of_all_stages
    << " " << (double)(stage_3) / sum_of_all_stages <<" " << (double)(stage_4) / sum_of_all_stages
    // << " " << (double)(stage_5) / sum_of_all_stages <<" " << (double)(stage_6) / sum_of_all_stages
    // << " " << (double)(stage_7) / sum_of_all_stages 
    << endl; 
    cudaMemcpy(h_IO_count, IO_count, num_of_query_points * sizeof(size_t), cudaMemcpyDeviceToHost);
    size_t avg_IO = 0;
    for(int i = 0; i < num_of_query_points; i++){
        avg_IO += h_IO_count[i];
    }
    cout << "avgrage IO: " << double(avg_IO) / double(num_of_query_points) << endl;
    // cudaMemcpy(h_blockTimes, d_blockTimes, num_of_query_points * sizeof(unsigned long long),
    // cudaMemcpyDeviceToHost); unsigned long long max_clock = 0; int max_id;
    // unsigned long long avg_clock = 0;
    // for(int i = 0; i < num_of_query_points; i++){
    // 	avg_clock += h_blockTimes[i];
    // 	if(h_blockTimes[i] > max_clock){
    // 		max_clock = h_blockTimes[i];
    // 		max_id = i;
    // 	}
    // }
    // printf("Avg clock: %lld, Max clock: %lld, Max id: %d\n",
    // avg_clock/num_of_query_points, max_clock, max_id);
}