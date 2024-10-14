#include <cooperative_groups.h>
#include "nsw_graph_operations.cuh"
//#include "../../RVQ/RVQ.cuh"
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "../graph_kernel_operation/kernel_local_graph_construction.cuh"
#include "../graph_kernel_operation/kernel_local_neighbors_sort_nsw.cuh"
#include "../graph_kernel_operation/kernel_local_graph_mergence_nsw.cuh"
#include "../graph_kernel_operation/kernel_global_edge_sort.cuh"
#include "../graph_kernel_operation/kernel_aggregate_forward_edges.cuh"
#include "../graph_kernel_operation/kernel_search_nsw.cuh"
#include "../graph_kernel_operation/cagra.cuh"
#include "../graph_kernel_operation/test_search.cuh"
// #include "../graph_kernel_operation/kernel_detourable_route_count.cuh"
// #include "../graph_kernel_operation/kernel_detourable_route_sort.cuh"
// #include "../graph_kernel_operation/kernel_reversed_detourable_route_sort.cuh"
// #include "../graph_kernel_operation/kernel_reversed_detourable_route_build.cuh"
#include "auto_tune_bloom.h"

cudaError_t error_check(cudaError_t error_code, int line)
{
        if (error_code != cudaSuccess)
        {
                printf("line: %d, error_code: %d, error_name: %s, error_description: %s\n",
                                line, error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));
        }
        return error_code;
}


__global__
void ConvertNeighborstoGraph(int* d_graph, KernelPair<float, int>* d_neighbors, int total_num_of_points, int offset_shift, int num_of_iterations, float* d_distance){
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = (1 << offset_shift) / 2;

    for (int i = 0; i < num_of_iterations; i++) {
    	int crt_point_id = i * base + b_id;
    	KernelPair<float, int>* crt_neighbors = d_neighbors + (crt_point_id << offset_shift);
    	int* crt_graph = d_graph + crt_point_id * num_of_final_neighbors;
		float* crt_distance = d_distance + crt_point_id * num_of_final_neighbors;
    	if (crt_point_id < total_num_of_points) {
        	for (int j = 0; j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int unroll_t_id = t_id + size_of_warp * j;
	
        	    if (unroll_t_id < num_of_final_neighbors) {
        	        crt_graph[unroll_t_id] = crt_neighbors[unroll_t_id].second;
					crt_distance[unroll_t_id] = crt_neighbors[unroll_t_id].first;
        	    }
        	}
    	}
    }

}

__global__
void LoadFirstSubgraph(pair<float, int>* first_subgraph, KernelPair<float, int>* d_first_subgraph, int num_of_copied_edges) {
	int t_id = threadIdx.x;
	int b_id = blockIdx.x;
	int global_t_id = b_id * blockDim.x + t_id;

	if (global_t_id < num_of_copied_edges) {
		d_first_subgraph[global_t_id].first = first_subgraph[global_t_id].first;
		d_first_subgraph[global_t_id].second = first_subgraph[global_t_id].second;
	}
}

void NSWGraphOperations::LocalGraphConstructionBruteForce(float* h_data, int offset_shift, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors,
										int num_of_batches, int num_of_points_one_batch, float* &d_data, KernelPair<float, int>* &d_neighbors,
										KernelPair<float, int>* &d_neighbors_backup){

	error_check( cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point), __LINE__);

	error_check( cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice), __LINE__);

	error_check( cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift)), __LINE__);
	error_check(cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift)), __LINE__);
	int num_of_batches_tmp = 2000;
	for(int i = 0; i * num_of_batches_tmp < num_of_batches; i++){
		KernelPair<float, int>* d_distance_matrix;
		error_check( cudaMalloc(&d_distance_matrix, num_of_batches_tmp * num_of_points_one_batch * num_of_points_one_batch * sizeof(KernelPair<float, int>)), __LINE__);
		
		DistanceMatrixComputation<<<num_of_batches_tmp, 32>>>(d_data, total_num_of_points, num_of_points_one_batch, d_distance_matrix, i);
		error_check(cudaGetLastError(), __LINE__);
		
		SortNeighborsonLocalGraph<<<num_of_points_one_batch, 32, 2 * num_of_initial_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors, d_neighbors_backup, total_num_of_points, 
																																	d_data, 
																																	num_of_points_one_batch,  
																																	num_of_initial_neighbors,
																																	offset_shift, 
																																	d_distance_matrix, i, num_of_batches_tmp);
		error_check(cudaGetLastError(), __LINE__);
																														
		
		cudaFree(d_distance_matrix);
	}
}
__global__
void initializeReversedGraph(int* d_reversed_graph, int* d_num_of_reversed_detourable_route, int num_of_initial_neighbors, int total_num_of_points, int num_of_iterations){
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    for (int i = 0; i < num_of_iterations; i++) {
    	int crt_point_id = i * base + b_id;
    	int* crt_graph = d_reversed_graph + crt_point_id * num_of_initial_neighbors;
		int* crt_num_of_reversed_detourable_route = d_num_of_reversed_detourable_route + crt_point_id * num_of_initial_neighbors;
    	if (crt_point_id < total_num_of_points) {
        	for (int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int unroll_t_id = t_id + size_of_warp * j;
	
        	    if (unroll_t_id < num_of_initial_neighbors) {
        	        crt_graph[unroll_t_id] = total_num_of_points;
					crt_num_of_reversed_detourable_route[unroll_t_id] = MAX;
        	    }
        	}
    	}
    }
}

void NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(float* d_data, int* &h_graph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_initial_neighbors, int num_of_batches, 
												int num_of_points_one_batch, KernelPair<float, int>* d_neighbors, KernelPair<float, int>* d_neighbors_backup,
												int num_of_final_neighbors, int num_of_candidates, pair<float, int>* first_subgraph, float* h_distance){

	int* d_graph;

	cudaMalloc(&d_graph, sizeof(int) * (total_num_of_points << offset_shift));

	Edge* 			d_edge_all_blocks;
	int* 			d_flag_all_blocks;
	int 			num_of_forward_edges;

	unsigned long long int* h_block_recorder;
	unsigned long long int* d_block_recorder;

	num_of_forward_edges = pow(2.0, ceil(log(num_of_points_one_batch) / log(2))) * num_of_initial_neighbors;

	cudaMalloc(&d_edge_all_blocks, num_of_forward_edges * sizeof(Edge));

	cudaMalloc(&d_flag_all_blocks, (num_of_forward_edges + 1) * sizeof(int));

	cudaMallocHost(&h_block_recorder, num_of_points_one_batch * sizeof(unsigned long long int));
	cudaMalloc(&d_block_recorder, num_of_points_one_batch * sizeof(unsigned long long int));

	pair<float, int>* d_first_subgraph;
	cudaMalloc(&d_first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(pair<float, int>));
	cudaMemcpy(d_first_subgraph, first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(pair<float, int>), cudaMemcpyHostToDevice);
	
	LoadFirstSubgraph<<<num_of_points_one_batch, num_of_final_neighbors>>>(d_first_subgraph, d_neighbors, num_of_points_one_batch * num_of_final_neighbors);

	for (int i = 1; i < num_of_batches; i++) {
		
		LocalGraphMergence<<<num_of_points_one_batch, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(
																									d_neighbors, d_neighbors_backup, total_num_of_points, d_data,
																									d_edge_all_blocks, i, num_of_points_one_batch, num_of_final_neighbors + num_of_candidates, 
																									num_of_final_neighbors, num_of_candidates, num_of_initial_neighbors, offset_shift, d_block_recorder);

		
		dim3 grid_of_kernel_edge_sort(num_of_forward_edges / 128, 1, 1);
		dim3 block_of_kernel_edge_sort(128, 1, 1);
		
		int num_of_valid_edges = num_of_points_one_batch * num_of_initial_neighbors;
		if (i == num_of_batches - 1) {
			num_of_valid_edges = (total_num_of_points - (num_of_batches - 1) * num_of_points_one_batch) * num_of_initial_neighbors;
		}

		void *kernel_args[] = {
			(void *)&d_neighbors, (void *)&d_edge_all_blocks, (void *)&d_flag_all_blocks, (void *)&num_of_forward_edges, 
			(void *)&num_of_valid_edges, (void *)&total_num_of_points
		};

		//sort for edges
		cudaLaunchCooperativeKernel((void *)GlobalEdgesSort, grid_of_kernel_edge_sort, block_of_kernel_edge_sort, kernel_args, 0);

		int num_of_types_valid_edges = 0;
		cudaMemcpy(&num_of_types_valid_edges, d_flag_all_blocks + num_of_forward_edges, sizeof(int), cudaMemcpyDeviceToHost);

		AggragateForwardEdges<<<num_of_types_valid_edges, 32, 2 * num_of_final_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors, d_edge_all_blocks, d_flag_all_blocks, 
																																total_num_of_points, num_of_final_neighbors, offset_shift);
	}

	int num_of_blocks = 10000;
	int num_of_iterations = (total_num_of_points + num_of_blocks - 1) / num_of_blocks;
	float* d_distance;
	cudaMalloc(&d_distance, sizeof(float) * (total_num_of_points << offset_shift));
	ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph, d_neighbors, total_num_of_points, offset_shift, num_of_iterations, d_distance);
	// cudaMemcpy(h_distance, d_distance, sizeof(float) * (total_num_of_points << offset_shift), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_graph, d_graph, sizeof(int) * total_num_of_points * num_of_initial_neighbors, cudaMemcpyDeviceToHost);
	cudaFree(d_first_subgraph);
	cudaFree(d_edge_all_blocks);
	cudaFree(d_flag_all_blocks);
	cudaFree(d_neighbors);
	cudaFree(d_neighbors_backup);
	cudaFree(d_data);
	cudaFree(d_distance);
	// // Graph optimization
	// printf("Graph optimization\n");
	// int* d_num_of_detourable_route;
	// cudaMalloc(&d_num_of_detourable_route, sizeof(int) * (total_num_of_points << offset_shift));
	// cudaMemset(d_num_of_detourable_route, 0, sizeof(int) * (total_num_of_points << offset_shift));
	// countDetourableRoute<<<num_of_blocks, 32>>>(d_graph, d_num_of_detourable_route, offset_shift, num_of_iterations, total_num_of_points);
	// //reordering and pruning
	// int* d_reversed_graph;
	// cudaMalloc(&d_reversed_graph, sizeof(int) * total_num_of_points * num_of_initial_neighbors);
	// int* d_num_of_reversed_neighbors;
	// cudaMalloc(&d_num_of_reversed_neighbors, sizeof(int) * total_num_of_points);
	// cudaMemset(d_num_of_reversed_neighbors, 0, sizeof(int) * total_num_of_points);
	// int* d_num_of_reversed_detourable_route;
	// cudaMalloc(&d_num_of_reversed_detourable_route, sizeof(int) * total_num_of_points * num_of_initial_neighbors);
	// initializeReversedGraph<<<num_of_blocks, 32>>>(d_reversed_graph, d_num_of_reversed_detourable_route, num_of_initial_neighbors, total_num_of_points, num_of_iterations);
	// constexpr int WARP_SIZE = 32;
	// constexpr int NumWarpQ = 32;
	// constexpr int NumThreadQ = 2;
	
	// sortDetourableRoute<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<num_of_blocks, 32>>>(d_graph, d_num_of_detourable_route, offset_shift, num_of_initial_neighbors, num_of_iterations, total_num_of_points);
	// //build and merge reversed graph
	// buildReversedDetourableRoute<<<num_of_blocks, 32>>>(d_graph, d_num_of_detourable_route, d_reversed_graph, d_num_of_reversed_neighbors,  d_num_of_reversed_detourable_route, 
    //                                          			offset_shift, num_of_initial_neighbors, num_of_iterations, total_num_of_points);
	
	// sortReversedDetourableRoute<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<num_of_blocks, 32>>>(d_graph, d_reversed_graph, d_num_of_reversed_neighbors, d_num_of_reversed_detourable_route, 
    //                                         														offset_shift, num_of_initial_neighbors, num_of_iterations, total_num_of_points);
	// cudaMemcpy(h_graph, d_reversed_graph, sizeof(int) * total_num_of_points * num_of_initial_neighbors, cudaMemcpyDeviceToHost);
	// cudaFree(d_graph);
	// cudaFree(d_num_of_detourable_route);
	// cudaFree(d_reversed_graph);
	// cudaFree(d_num_of_reversed_neighbors);
	// cudaFree(d_num_of_reversed_detourable_route);
}

__global__
void zeroCount(int* d_enter_cluster,int* d_rvq_index_sizes,int* d_num_of_zero_query, int* d_max_cluster){
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int cluster_id = d_enter_cluster[bid];
	int cluster_size = d_rvq_index_sizes[cluster_id];
	if(tid == 0){
		//printf("%d ",cluster_size);
		if(cluster_size == 0){
			atomicAdd(&d_num_of_zero_query[0], 1);
		}
		atomicMax(&d_max_cluster[0], cluster_size);
	}
}

void readGraph(string path, int* graph, int num_of_subgraph, int num_of_candidates, int num_of_neighbors, int total_num_of_points,int* pre_fix_of_subgraph_size){
	for(int i = 0; i < num_of_subgraph; i++){
		ostringstream graph_filename;
		graph_filename <<"finalGraph"<<std::setw(4)<<std::setfill('0')<<i<<"_"<<32<<"_"<<
		16<<"_"<<to_string(total_num_of_points/1000000)<<"M"<<".nsw";
		string main_graph_path = path + graph_filename.str();
		ifstream in_descriptor(main_graph_path, std::ios::binary);
		in_descriptor.read((char*)(graph + pre_fix_of_subgraph_size[i] * num_of_neighbors), sizeof(int) * (pre_fix_of_subgraph_size[i + 1] - pre_fix_of_subgraph_size[i]) * num_of_neighbors);
		in_descriptor.close();
	}
}


void readCagraGraph(string path, int n, int dim, int* graph){
    ifstream in_descriptor(path, std::ios::binary);
    
    if (!in_descriptor.is_open()) {
        exit(1);
    }

    in_descriptor.seekg(790, std::ios::beg);
    in_descriptor.read((char*)(graph), (n << dim) * sizeof(int));
	// int data_size = 800;
	// char neighbors[800];
	// in_descriptor.read((char*)(neighbors), data_size * sizeof(char));
    // for(int i = 0; i < data_size; i++){
    //     cout<<neighbors[i];
    // }
    // cout<<endl;
    // in_descriptor.close();
}


__global__
void queryCopy(float* d_query, int dim, int copy_id){
	int b_id = blockIdx.x;
	int t_id = threadIdx.x;
	int size_of_block = blockDim.x;
	if(b_id == copy_id) return;
	float* copy_from = d_query + dim * copy_id;
	float* copy_to = d_query + dim * b_id;
	for(int i = 0; i < (dim + size_of_block - 1) / size_of_block; i++){
		int unrollid = t_id + i * size_of_block;
		if(unrollid < dim){
			copy_to[unrollid] = copy_from[unrollid];
		}
	}
}

void NSWGraphOperations::Search(float* d_data, float* d_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, 
					int offset_shift, int num_of_topk, int num_of_candidates, int num_of_explored_points, int* d_enter_cluster, GPUIndex* d_rvq_index, Timer* &graphSearch, int search_width) {

	// float* d_data;
	// cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
	// cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);
	// string base_path = "/home/ErHa/GANNS_Res/subdata/";
	// string pre_fix_of_subgraph_size_path = base_path + "pre_fix_of_subgraph_size.bin";
    // std::ifstream in(pre_fix_of_subgraph_size_path, std::ios::binary);
    // int num_of_subgraph;
    // in.read(reinterpret_cast<char*>(&num_of_subgraph), sizeof(int));
    // int* pre_fix_of_subgraph_size = new int[num_of_subgraph + 1];
    // in.read(reinterpret_cast<char*>(pre_fix_of_subgraph_size), (num_of_subgraph + 1) * sizeof(int));
    // in.close();
	// string new_index_of_data_path = base_path + "new_index_of_data.bin";
	// std::ifstream inn(new_index_of_data_path, std::ios::binary);
    // int num_of_index;
    // inn.read(reinterpret_cast<char*>(&num_of_index), sizeof(int));
    // int* new_index_of_data = new int[num_of_index];
    // inn.read(reinterpret_cast<char*>(new_index_of_data), num_of_index * sizeof(int));
    // inn.close();
	// readGraph("/home/ErHa/GANNS_Res/finalgraph/", h_graph, num_of_subgraph, num_of_candidates, (1 << offset_shift), total_num_of_points,pre_fix_of_subgraph_size);
	
	// int* d_pre_fix_of_subgraph_size;
	// cudaMalloc(&d_pre_fix_of_subgraph_size, sizeof(int) * (num_of_subgraph + 1));
	// cudaMemcpy(d_pre_fix_of_subgraph_size, pre_fix_of_subgraph_size, sizeof(int) * (num_of_subgraph + 1), cudaMemcpyHostToDevice);

	// int* old_index_of_data = new int[num_of_index];
	// for(int i = 0; i < num_of_index; i++){
	// 	old_index_of_data[new_index_of_data[i]] = i;
	// }
	// int* d_new_index_of_data;
	// cudaMalloc(&d_new_index_of_data, sizeof(int) * num_of_index);
	// cudaMemcpy(d_new_index_of_data, new_index_of_data, sizeof(int) * num_of_index, cudaMemcpyHostToDevice);

	// int* d_old_index_of_data;
	// cudaMalloc(&d_old_index_of_data, sizeof(int) * num_of_index);
	// cudaMemcpy(d_old_index_of_data, old_index_of_data, sizeof(int) * num_of_index, cudaMemcpyHostToDevice);

	// readCagraGraph("/home/szr/test/cagra_test/sift1m_degree8_index.bin", total_num_of_points, offset_shift, h_graph);
	int* d_graph;
	cudaMalloc(&d_graph, sizeof(int) * (total_num_of_points << offset_shift));
	cudaMemcpy(d_graph, h_graph, sizeof(int) * (total_num_of_points << offset_shift), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	graphSearch[1].Start();
	int* d_result;
	cudaMalloc(&d_result, sizeof(int) * (num_of_topk * num_of_query_points));
	graphSearch[1].Stop();


	// unsigned long long* h_time_breakdown;
	// unsigned long long* d_time_breakdown;
	// int num_of_phases = 7;
	// cudaMallocHost(&h_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	// cudaMalloc(&d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	// cudaMemset(d_time_breakdown, 0, num_of_query_points * num_of_phases * sizeof(unsigned long long));

	// int* h_num_of_zero_query = new int[1];
	// int* d_num_of_zero_query;
	// cudaMalloc(&d_num_of_zero_query, sizeof(int));
	// cudaMemset(d_num_of_zero_query, 0, sizeof(int));

	// int* h_max_cluster = new int[1];
	// int* d_max_cluster;
	// cudaMalloc(&d_max_cluster, sizeof(int));
	// cudaMemset(d_max_cluster, 0, sizeof(int));

	// int h_count[num_of_query_points] = {0};
	// int* d_count;
	// cudaMalloc(&d_count, sizeof(int) * num_of_query_points);
    // cudaMemcpy(d_count, &h_count, sizeof(int) * num_of_query_points, cudaMemcpyHostToDevice);

	// int h_zero_count[num_of_query_points] = {0};
	// int* d_zero_count;
	// cudaMalloc(&d_zero_count, sizeof(int) * num_of_query_points);
    // cudaMemcpy(d_zero_count, &h_zero_count, sizeof(int) * num_of_query_points, cudaMemcpyHostToDevice);

	// int h_iter[num_of_query_points] = {0};
	// int* d_iter;
	// cudaMalloc(&d_iter, sizeof(int) * num_of_query_points);
    // cudaMemcpy(d_iter, &h_count, sizeof(int) * num_of_query_points, cudaMemcpyHostToDevice);
	// zeroCount<<<num_of_query_points, 1>>>(d_enter_cluster, d_rvq_index->sizes, d_num_of_zero_query, d_max_cluster);
	// queryCopy<<<num_of_query_points, dim_of_point>>>(d_query, dim_of_point, 4136);
	// cudaDeviceSynchronize();
	// int T_size = max(128, (1 << offset_shift));
	float epsilon = 1;
	//int search_width = 8;
	int hash_len, bit, hash;
	hash_parameter(num_of_candidates, hash_len, bit, hash);
	printf("bit %d %d hash %d\n",hash_len, bit, hash);
	// unsigned long long* h_blockTimes = new unsigned long long[num_of_query_points]; 
    // unsigned long long* d_blockTimes; 
	// cudaMalloc(&d_blockTimes, num_of_query_points * sizeof(unsigned long long));
	// cudaMemset(d_blockTimes, 0, num_of_query_points * sizeof(unsigned long long));
	constexpr int WARP_SIZE = 32;
	constexpr int NumWarpQ = 32;
	constexpr int NumThreadQ = 1;
	cudaDeviceSynchronize();
	graphSearch[2].Start();
	
	cagra<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<num_of_query_points, 32, ((search_width << offset_shift) + num_of_candidates) * sizeof(KernelPair<float, int>)>>> 
													(d_data, d_query, d_result, d_graph, total_num_of_points, 
													offset_shift, num_of_candidates, num_of_topk, num_of_explored_points, search_width,  
													d_enter_cluster, d_rvq_index->indices, d_rvq_index->sizes);
	cudaDeviceSynchronize();
	graphSearch[2].Stop();
	error_check(cudaGetLastError(), __LINE__);

	graphSearch[3].Start();
	cudaMemcpy(h_result, d_result, sizeof(int) * num_of_query_points * num_of_topk, cudaMemcpyDeviceToHost);
	graphSearch[3].Stop();
	cudaDeviceSynchronize();
	for(int i = 0; i < num_of_query_points; i++){
		int flag = 0;
		for(int l = 0; l < num_of_topk; l++){
			// printf("%d ", h_result[i * num_of_topk + l]);
			for(int k = l ; k < num_of_topk; k++){
				if(l == k || h_result[i * num_of_topk + l] == total_num_of_points) continue;
				if(h_result[i * num_of_topk + l] == h_result[i * num_of_topk + k]){
					printf("%d %d %d\n", i, l, h_result[i * num_of_topk + l]);
					// for(int j = 0; j < num_of_topk; j++){
					// 	printf("%d ", h_result[i * num_of_topk + j]);
					// }
					// printf("\n");
					flag = 1;
					break;
				}
			}
			if(flag == 1) break;
		}
		// printf("\n");
	}

	// cudaMemcpy(&h_count, d_count, sizeof(int) * num_of_query_points, cudaMemcpyDeviceToHost);
	// cudaMemcpy(&h_zero_count, d_zero_count, sizeof(int) * num_of_query_points, cudaMemcpyDeviceToHost);
	// cudaMemcpy(&h_iter, d_iter, sizeof(int) * num_of_query_points, cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_num_of_zero_query,d_num_of_zero_query, sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_max_cluster,d_max_cluster, sizeof(int), cudaMemcpyDeviceToHost);
	// int h_count_sum = 0;
	// int h_zero_count_sum=0;
	// int h_iter_sum=0;
	// int max_iter = 0;
	// int max_iter_id;
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
    // printf("Number of query without zero cluster computation: %d\n", h_count_sum);
	// printf("Number of query with zero cluster: %d\n", h_num_of_zero_query[0]);
	// printf("Number of query with zero cluster computation: %d\n", h_zero_count_sum);
	// printf("Number of avg iter: %d\n", h_iter_sum / num_of_query_points);
	// printf("Number of max iter: %d, id: %d\n", max_iter, max_iter_id);

	

	// cudaFree(d_graph);
	// cudaFree(d_data);
	// cudaFree(d_result);
	// cudaFree(d_count);
	
	// cudaMemcpy(h_time_breakdown, d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	// unsigned long long stage_1 = 0;
	// unsigned long long stage_2 = 0;
	// unsigned long long stage_3 = 0;
	// unsigned long long stage_4 = 0;
	// unsigned long long stage_5 = 0;
	// unsigned long long stage_6 = 0;
	// unsigned long long stage_7 = 0;

	// for (int i = 0; i < num_of_query_points; i++) {
	// 	stage_1	+= h_time_breakdown[i * num_of_phases];
	// 	stage_2	+= h_time_breakdown[i * num_of_phases + 1];
	// 	stage_3	+= h_time_breakdown[i * num_of_phases + 2];
	// 	stage_4	+= h_time_breakdown[i * num_of_phases + 3];
	// 	stage_5	+= h_time_breakdown[i * num_of_phases + 4];
	// 	stage_6	+= h_time_breakdown[i * num_of_phases + 5];
	// 	stage_7 += h_time_breakdown[i * num_of_phases + 6];
	// }
	// unsigned long long sum_of_all_stages = stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6 + stage_7;
	// cout << "stages percentage: " << (double)(stage_1) / sum_of_all_stages << " "
	// 								<< (double)(stage_2) / sum_of_all_stages << " "
	// 								<< (double)(stage_3) / sum_of_all_stages << " "
	// 								<< (double)(stage_4) / sum_of_all_stages << " "
	// 								<< (double)(stage_5) / sum_of_all_stages << " "
	// 								<< (double)(stage_6) / sum_of_all_stages << " "
	// 								<< (double)(stage_7) / sum_of_all_stages <<endl;
	// cout << "stages time: " << (double)(stage_1)<< " "
	// 								<< (double)(stage_2)<< " "
	// 								<< (double)(stage_3)<< " "
	// 								<< (double)(stage_4)<< " "
	// 								<< (double)(stage_5)<< " "
	// 								<< (double)(stage_6)<< " "
	// 								<< (double)(stage_7)<<endl;
	// cudaMemcpy(h_blockTimes, d_blockTimes, num_of_query_points * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	// unsigned long long max_clock = 0;
	// int max_id;
	// unsigned long long avg_clock = 0;
	// for(int i = 0; i < num_of_query_points; i++){
	// 	avg_clock += h_blockTimes[i];
	// 	if(h_blockTimes[i] > max_clock){
	// 		max_clock = h_blockTimes[i];
	// 		max_id = i;
	// 	}
	// }
	// printf("Avg clock: %lld, Max clock: %lld, Max id: %d\n", avg_clock/num_of_query_points, max_clock, max_id);

}