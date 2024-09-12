#include "subgraph_operation.cuh"
#include "kernel_subgraph_mergence_nsw.cuh"

template <typename T> 
void readGraphData(string path, int dim, int num, T* data){
	//data = new T[dim * num];
	ifstream in_descriptor(path, std::ios::binary);
	in_descriptor.read((char*)data, sizeof(T) * num * dim);
	in_descriptor.close();
}

void readData(string path, int dim, int num, float* data){
    ifstream in_descriptor(path, std::ios::binary);
    
    if (!in_descriptor.is_open()) {
        exit(1);
    }
    in_descriptor.seekg(0, std::ios::beg);
    for (int i = 0; i < num; i++) {
        unsigned char tmp_data[dim];
        in_descriptor.seekg(4, std::ios::cur);
        //in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_);
        in_descriptor.read((char*)(tmp_data), dim);
        for(int l = 0;l < dim; l++){
            data[i * dim + l] = float(tmp_data[l]);
        }
    }

    in_descriptor.close();
}

void subgraph::readSubgraph(int index_of_subgraph, int num_of_subgraph_points, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_final_neighbors, 
                       int num_of_initial_neighbors, int num_of_candidates, KernelPair<float, int>* d_graph, float* d_data, bool final_graph){
    float* h_data;
    h_data = new float[dim_of_point * num_of_subgraph_points];
    ostringstream data_filename;
    data_filename << "subData" << std::setw(4) << std::setfill('0') << index_of_subgraph << ".bin";
    string data_path = base_path + data_filename.str();
    readData(data_path, dim_of_point, num_of_subgraph_points, h_data);
    cudaMemcpy(d_data, h_data, sizeof(float) * dim_of_point * num_of_subgraph_points, cudaMemcpyHostToDevice);
    
    // if(!final_graph){
    //     for(int l = 0; l < 10; l++){
    //         for(int j = 0; j < dim_of_point; j++){
    //             printf("%f ", h_data[l * dim_of_point + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    int* h_main_graph;
    h_main_graph = new int[num_of_final_neighbors * num_of_subgraph_points];
    ostringstream main_graph_filename;
    string main_graph_path;
    if(final_graph){
        main_graph_filename <<"finalGraph"<<std::setw(4)<<std::setfill('0')<<index_of_subgraph<<"_"<<num_of_candidates<<"_"<<
        num_of_initial_neighbors<<"_"<<to_string(total_num_of_points/1000000)<<"M"<<".nsw";
        main_graph_path = final_graph_path + main_graph_filename.str();
    }else{
        main_graph_filename <<"subGraph"<<std::setw(4)<<std::setfill('0')<<index_of_subgraph<<"_"<<num_of_candidates<<"_"<<
        num_of_initial_neighbors<<"_"<<to_string(total_num_of_points/1000000)<<"M"<<".nsw";
        main_graph_path = graph_path + main_graph_filename.str();
    }
    readGraphData(main_graph_path, num_of_final_neighbors, num_of_subgraph_points, h_main_graph);

    float* h_main_dis;
    h_main_dis = new float[num_of_final_neighbors * num_of_subgraph_points];
    ostringstream main_dis_filename;
    main_dis_filename <<"subGraphDis"<<std::setw(4)<<std::setfill('0')<<index_of_subgraph;
    string main_dis_path = graph_path + main_dis_filename.str();
    readGraphData(main_dis_path, num_of_final_neighbors, num_of_subgraph_points, h_main_dis);
    
    KernelPair<float, int>* h_graph;
    h_graph = new KernelPair<float, int>[num_of_final_neighbors * num_of_subgraph_points];
    for(int l = 0; l < num_of_subgraph_points; l++){
        for(int k = 0; k < num_of_final_neighbors; k++){
            h_graph[l * num_of_final_neighbors + k].first = h_main_dis[l * num_of_final_neighbors + k];
            h_graph[l * num_of_final_neighbors + k].second = h_main_graph[l * num_of_final_neighbors + k];
        }
    }
    cudaMemcpy(d_graph, h_graph, sizeof(KernelPair<float, int>) * num_of_final_neighbors * num_of_subgraph_points, cudaMemcpyHostToDevice);

    delete[] h_data;
    delete[] h_main_graph;
    delete[] h_main_dis;
    delete[] h_graph;
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

void subgraph::saveFinalGraph(string final_graph_filename, int index_of_subgraph, int num_of_subgraph_points, int num_of_final_neighbors, int offset_shift, KernelPair<float, int>* d_neighbors){
    int* h_graph;
    h_graph = new int[num_of_subgraph_points * num_of_final_neighbors];
    int* d_graph;
    cudaMalloc(&d_graph, num_of_subgraph_points * num_of_final_neighbors * sizeof(int));
    int num_of_blocks = 10000;
	int num_of_iterations = (num_of_subgraph_points + num_of_blocks - 1) / num_of_blocks;
	ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph, d_neighbors, num_of_subgraph_points, offset_shift, num_of_iterations);
    cudaMemcpy(h_graph, d_graph, sizeof(int) * num_of_final_neighbors * num_of_subgraph_points, cudaMemcpyDeviceToHost);

    string save_path = final_graph_path + final_graph_filename;
    ofstream out_graph(save_path, std::ios::binary);
    out_graph.write((char*)h_graph, sizeof(int) * num_of_final_neighbors * num_of_subgraph_points);
    out_graph.close();

    cudaFree(d_graph);
    delete[] h_graph;

}

void subgraph::subgraphMerge(int num_of_subgraph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_final_neighbors, 
                       int num_of_initial_neighbors, int num_of_candidates, int* pre_fix_of_subgraph_size){
    for(int i = 0; i < num_of_subgraph; i++){
        //读取主图i
        printf("%d subgraph merge...\n", i);
		int num_of_main_graph_points = pre_fix_of_subgraph_size[i + 1] - pre_fix_of_subgraph_size[i];
        
        int num_of_points_one_batch = 10000;
        int num_of_batches;
        num_of_batches = (num_of_main_graph_points + num_of_points_one_batch - 1) / num_of_points_one_batch;
        num_of_points_one_batch = (num_of_main_graph_points + num_of_batches - 1) / num_of_batches;
        Edge* d_edge_all_blocks;
        //int*  d_flag_all_blocks;
        //int   num_of_forward_edges;
        //num_of_forward_edges = pow(2.0, ceil(log(num_of_points_one_batch) / log(2))) * num_of_final_neighbors;
        // cudaMalloc(&d_edge_all_blocks, num_of_forward_edges * sizeof(Edge));
        // cudaMalloc(&d_flag_all_blocks, (num_of_forward_edges + 1) * sizeof(int));
		
        float* d_main_data;
        cudaMalloc(&d_main_data, sizeof(float) * dim_of_point * num_of_main_graph_points);

		KernelPair<float, int>* d_main_graph;
        cudaMalloc(&d_main_graph, sizeof(KernelPair<float, int>) * num_of_final_neighbors * num_of_main_graph_points);

		readSubgraph(i, num_of_main_graph_points, total_num_of_points, dim_of_point, offset_shift, num_of_final_neighbors, num_of_initial_neighbors, 
                     num_of_candidates, d_main_graph, d_main_data, 1);
        
        for(int l = 0; l < num_of_subgraph; l++){
            //读取要和主图合并的图l
            if(l == i) continue;
            int num_of_sub_graph_points = pre_fix_of_subgraph_size[l + 1] - pre_fix_of_subgraph_size[l];
            
            float* d_sub_data;
            cudaMalloc(&d_sub_data, sizeof(float) * dim_of_point * num_of_sub_graph_points);

            KernelPair<float, int>* d_sub_graph;
            cudaMalloc(&d_sub_graph, sizeof(KernelPair<float, int>) * num_of_final_neighbors * num_of_sub_graph_points);

            readSubgraph(l, num_of_sub_graph_points, total_num_of_points, dim_of_point, offset_shift, num_of_final_neighbors, num_of_initial_neighbors, 
                         num_of_candidates, d_sub_graph, d_sub_data, 0);

            //合并
            for(int k = 0; k < num_of_batches; k++){
                SubGraphMergence<<<num_of_points_one_batch, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>
                                (d_main_graph, d_sub_graph, num_of_main_graph_points, num_of_sub_graph_points, total_num_of_points, pre_fix_of_subgraph_size[i],
                                 pre_fix_of_subgraph_size[l], d_main_data, d_sub_data, d_edge_all_blocks, k, num_of_points_one_batch, num_of_final_neighbors + num_of_candidates,
                                 num_of_final_neighbors, num_of_candidates, num_of_initial_neighbors, offset_shift);
            }
            cudaFree(d_sub_data);
            cudaFree(d_sub_graph);
        }
        ostringstream final_graph_filename;
        final_graph_filename <<"finalGraph"<<std::setw(4)<<std::setfill('0')<<i<<"_"<<num_of_candidates<<"_"<<
        num_of_initial_neighbors<<"_"<<to_string(total_num_of_points/1000000)<<"M"<<".nsw";
        saveFinalGraph(final_graph_filename.str(), i, num_of_main_graph_points, num_of_final_neighbors, offset_shift, d_main_graph);

        cudaFree(d_main_data);
        cudaFree(d_main_graph);
	}

}


void subgraph::subgraphBuild(int num_of_candidates, int num_of_initial_neighbors, int num_of_points){
    string pre_fix_of_subgraph_size_path = base_path + "pre_fix_of_subgraph_size.bin";
    std::ifstream in(pre_fix_of_subgraph_size_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }
    int num_of_subgraph;
    in.read(reinterpret_cast<char*>(&num_of_subgraph), sizeof(int));
    int* pre_fix_of_subgraph_size = new int[num_of_subgraph + 1];
    in.read(reinterpret_cast<char*>(pre_fix_of_subgraph_size), (num_of_subgraph + 1) * sizeof(int));
    in.close();
    // printf("There are %d subgraph\n",num_of_subgraph);
    // for(int i = 0; i < num_of_subgraph; i++){
    //     printf("Load subgraph %d data points...\n", i);
    //     std::ostringstream subgraph_data_filename;
    //     subgraph_data_filename << "subData" << std::setw(4) << std::setfill('0') << i << ".bin";
    //     string subgraph_data_path = base_path + subgraph_data_filename.str();

    //     int num_of_subgraph_points = pre_fix_of_subgraph_size[i + 1] - pre_fix_of_subgraph_size[i];
    //     Data* points = new Data(subgraph_data_path,num_of_subgraph_points);

    //     GraphWrapper* graph;
    //     graph = new NavigableSmallWorldGraphWithFixedDegree(points);
    //     printf("Construct %d subgraph ...\n", i);
    //     graph->Establishment(num_of_initial_neighbors, num_of_candidates);

    //     printf("Save %d subgraph ...\n", i);
    //     ostringstream subgraph_filename;
    //     subgraph_filename <<"subGraph"<<std::setw(4)<<std::setfill('0')<<i<<"_"<<to_string(num_of_candidates)<<"_"<<
    //     to_string(num_of_initial_neighbors)<<"_"<<to_string(num_of_points/1000000)<<"M"<<".nsw";
    //     string subgraph_path = graph_path + subgraph_filename.str();
    //     ostringstream final_graph_filename;
    //     final_graph_filename <<"finalGraph"<<std::setw(4)<<std::setfill('0')<<i<<"_"<<num_of_candidates<<"_"<<
    //     num_of_initial_neighbors<<"_"<<to_string(num_of_points/1000000)<<"M"<<".nsw";
    //     string final_path = final_graph_path + final_graph_filename.str();

    //     ostringstream subgraph_dis_filename;
    //     subgraph_dis_filename <<"subGraphDis"<<std::setw(4)<<std::setfill('0')<<i;
    //     string subgraph_dis_path = graph_path + subgraph_dis_filename.str();

    //     graph->Dump(subgraph_path, subgraph_dis_path, final_path, pre_fix_of_subgraph_size[i], num_of_points);
    // }
    subgraphMerge(num_of_subgraph, num_of_points, 128, 5, num_of_initial_neighbors * 2, num_of_initial_neighbors, num_of_candidates, pre_fix_of_subgraph_size);
}