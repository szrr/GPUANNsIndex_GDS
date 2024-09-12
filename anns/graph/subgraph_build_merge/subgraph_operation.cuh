#pragma once
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include "../graph_index/navigable_small_world.cuh"
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "../../common.h"
using namespace std;

class subgraph{
public:
    subgraph(string base_path_, string graph_path_, string final_graph_path_){
        base_path = base_path_;
        graph_path = graph_path_;
        final_graph_path = final_graph_path_;
    }

    void readSubgraph(int index_of_subgraph, int num_of_subgraph_points, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_final_neighbors, 
                       int num_of_initial_neighbors, int num_of_candidates, KernelPair<float, int>* d_graph, float* d_data, bool final_graph);

    void saveFinalGraph(string final_graph_filename, int index_of_subgraph, int num_of_subgraph_points, int num_of_final_neighbors, int offset_shift, KernelPair<float, int>* d_neighbors);

    void subgraphMerge(int num_of_subgraph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_final_neighbors, 
                       int num_of_initial_neighbors, int num_of_candidates, int* pre_fix_of_subgraph_size);

    void subgraphBuild(int num_of_candidates, int num_of_initial_neighbors, int num_of_points);

    string base_path;
    string graph_path;
    string final_graph_path;
};