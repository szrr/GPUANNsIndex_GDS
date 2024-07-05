
#include <cuda_runtime.h>
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
#include "../graph_kernel_operation/test.cuh"


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
void ConvertNeighborstoGraph(int* d_graph, KernelPair<float, int>* d_neighbors, int total_num_of_points, int offset_shift, int num_of_iterations){
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = 1 << offset_shift;

    for (int i = 0; i < num_of_iterations; i++) {
    	int crt_point_id = i * base + b_id;
    	KernelPair<float, int>* crt_neighbors = d_neighbors + (crt_point_id << offset_shift);
    	int* crt_graph = d_graph + (crt_point_id << offset_shift);

    	if (crt_point_id < total_num_of_points) {
        	for (int j = 0; j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; j++) {
        	    int unroll_t_id = t_id + size_of_warp * j;
	
        	    if (unroll_t_id < num_of_final_neighbors) {
        	        crt_graph[unroll_t_id] = crt_neighbors[unroll_t_id].second;
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
	//cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);

	error_check( cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice), __LINE__);
	//cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

	error_check( cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift)), __LINE__);
	error_check(cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift)), __LINE__);
	//cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift));
	//cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift));
	int num_of_batches_tmp = 2000;
	for(int i = 0; i * num_of_batches_tmp < num_of_batches; i++){
		KernelPair<float, int>* d_distance_matrix;
		error_check( cudaMalloc(&d_distance_matrix, num_of_batches_tmp * num_of_points_one_batch * num_of_points_one_batch * sizeof(KernelPair<float, int>)), __LINE__);
		//cudaMalloc(&d_distance_matrix, total_num_of_points * num_of_points_one_batch * sizeof(KernelPair<float, int>));
		
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
	
void NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(float* d_data, int* &h_graph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_initial_neighbors, int num_of_batches, 
												int num_of_points_one_batch, KernelPair<float, int>* d_neighbors, KernelPair<float, int>* d_neighbors_backup,
												int num_of_final_neighbors, int num_of_candidates, pair<float, int>* first_subgraph){

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

	ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph, d_neighbors, total_num_of_points, offset_shift, num_of_iterations);
	cudaMemcpy(h_graph, d_graph, sizeof(int) * (total_num_of_points << offset_shift), cudaMemcpyDeviceToHost);

	cudaFree(d_edge_all_blocks);
	cudaFree(d_flag_all_blocks);
	cudaFree(d_neighbors);
	cudaFree(d_neighbors_backup);
	cudaFree(d_graph);
	cudaFree(d_data);
}

void NSWGraphOperations::Search(float* h_data, float* d_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, 
					int offset_shift, int num_of_topk, int num_of_candidates, int num_of_explored_points, int* d_enter_cluster, GPUIndex* d_rvq_index, Timer* &graphSearch) {

	float* d_data;
	cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
	cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

	
	int* d_graph;
	cudaMalloc(&d_graph, sizeof(int) * (total_num_of_points << offset_shift));
	cudaMemcpy(d_graph, h_graph, sizeof(int) * (total_num_of_points << offset_shift), cudaMemcpyHostToDevice);

	graphSearch[1].Start();

	int* d_result;
	cudaMalloc(&d_result, sizeof(int) * num_of_query_points * num_of_topk);
	graphSearch[1].Stop();


	unsigned long long* h_time_breakdown;
	unsigned long long* d_time_breakdown;
	int num_of_phases = 6;
	cudaMallocHost(&h_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	cudaMalloc(&d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
	cudaMemset(d_time_breakdown, 0, num_of_query_points * num_of_phases * sizeof(unsigned long long));

	int h_count[num_of_query_points] = {0};
	int* d_count;
	cudaMalloc(&d_count, sizeof(int) * num_of_query_points);
    cudaMemcpy(d_count, &h_count, sizeof(int) * num_of_query_points, cudaMemcpyHostToDevice);

	int shared_mem_size = max(128, (1 << offset_shift));
	error_check(cudaGetLastError(), __LINE__);
	graphSearch[2].Start();
	//DistanceOfEntryPoints<<<num_of_query_points,32>>>(d_data, d_query,d_entry_points,d_enter_points_num,d_enter_points_num_sort);
	
	SearchDevice<<<num_of_query_points, 32, (shared_mem_size + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(d_data, d_query, d_result, d_graph, total_num_of_points, 
																														num_of_query_points, offset_shift, num_of_candidates, num_of_topk, 
																														num_of_explored_points, d_time_breakdown, d_enter_cluster, d_rvq_index->indices, 
																														 d_rvq_index->sizes, shared_mem_size, d_count);
	// test<<<num_of_query_points, 32, ((1 << offset_shift) + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(d_data, d_query, d_result, d_graph, total_num_of_points, 
	// 																													num_of_query_points, offset_shift, num_of_candidates, num_of_topk, 
	// 																													num_of_explored_points, d_time_breakdown, d_count, d_enter_points_num);																													
	cudaDeviceSynchronize();
	error_check(cudaGetLastError(), __LINE__);
	graphSearch[2].Stop();

	graphSearch[3].Start();
	cudaMemcpy(h_result, d_result, sizeof(int) * num_of_query_points * num_of_topk, cudaMemcpyDeviceToHost);
	graphSearch[3].Stop();

	cudaMemcpy(&h_count, d_count, sizeof(int) * num_of_query_points, cudaMemcpyDeviceToHost);
	int h_count_sum = 0;
	for(int i=0; i<num_of_query_points; i++){
		h_count_sum+=h_count[i];
	}
    printf("Number of computation: %d\n", h_count_sum);

	cudaFree(d_graph);
	cudaFree(d_data);
	cudaFree(d_rvq_index);
	cudaFree(d_enter_cluster);
	cudaFree(d_query);
	cudaFree(d_result);
	
	/*cudaMemcpy(h_time_breakdown, d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	unsigned long long stage_1 = 0;
	unsigned long long stage_2 = 0;
	unsigned long long stage_3 = 0;
	unsigned long long stage_4 = 0;
	unsigned long long stage_5 = 0;
	unsigned long long stage_6 = 0;

	for (int i = 0; i < num_of_query_points; i++) {
		stage_1	+= h_time_breakdown[i * num_of_phases];
		stage_2	+= h_time_breakdown[i * num_of_phases + 1];
		stage_3	+= h_time_breakdown[i * num_of_phases + 2];
		stage_4	+= h_time_breakdown[i * num_of_phases + 3];
		stage_5	+= h_time_breakdown[i * num_of_phases + 4];
		stage_6	+= h_time_breakdown[i * num_of_phases + 5];
	}

	unsigned long long sum_of_all_stages = stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6;
	cout << "stages percentage: " << (double)(stage_1) / sum_of_all_stages << " "
									<< (double)(stage_2) / sum_of_all_stages << " "
									<< (double)(stage_3) / sum_of_all_stages << " "
									<< (double)(stage_4) / sum_of_all_stages << " "
									<< (double)(stage_5) / sum_of_all_stages << " "
									<< (double)(stage_6) / sum_of_all_stages << endl;*/
}