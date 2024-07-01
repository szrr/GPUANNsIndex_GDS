#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "../../common.h"

using namespace std;

class NSWGraphOperations {
public:

	static void LocalGraphConstructionBruteForce(float* h_data, int offset_shift, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors,
											int num_of_batches, int num_of_points_one_batch, float* &d_data, KernelPair<float, int>* &d_neighbors,
											KernelPair<float, int>* &d_neighbors_backup);
	
	static void LocalGraphMergenceCoorperativeGroup(float* d_data, int* &h_graph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_initial_neighbors, int num_of_batches, 
														int num_of_points_one_batch, KernelPair<float, int>* d_neighbors, KernelPair<float, int>* d_neighbors_backup,
														int num_of_final_neighbors, int num_of_candidates, pair<float, int>* first_subgraph);

	static void Search(float* h_data, float* h_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, 
						int offset_shift, int num_of_topk, int num_of_candidates, int num_of_explored_points, vector<std::vector<int>> enterPoints,Timer* &graphSearch); 

};