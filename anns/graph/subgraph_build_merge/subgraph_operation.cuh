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

#include "kernel_subgraph_mergence_nsw.cuh"
#include "kernel_k_subgraph_merge.cuh"
#include "../graph_kernel_operation/bamWriteSSD.cuh"

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

    void subgraphMerge(int num_of_subgraph, int total_num_of_points, int num_of_neighbors, 
                       int k, int* pre_fix_of_subgraph_size, 
                       array_d_t<int> *ssd_graph, array_d_t<float> *ssd_distance, size_t* pre_fix_of_start_loc);

    void subgraphBuild(int num_of_candidates, int num_of_initial_neighbors, int num_of_points, int k);

    string base_path;
    string graph_path;
    string final_graph_path;
};

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

void subgraph::subgraphMerge(int num_of_subgraph, int total_num_of_points, int num_of_neighbors, 
                       int k, int* pre_fix_of_subgraph_size,
                       array_d_t<int> *ssd_graph, array_d_t<float> *ssd_distance, size_t* pre_fix_of_start_loc){
    // int *d_int_output;
    // float *d_float_output;
    // cudaMalloc((void **)&d_int_output, 128 * sizeof(int));
    // cudaMalloc((void **)&d_float_output, 128 * sizeof(float));
    // read_graph_kernel<<<1, 64>>>(ssd_graph, d_int_output);

    // read_data_kernel<<<1, 64>>>(ssd_distance, d_float_output);
    size_t total_k_num_of_points = size_t(total_num_of_points) * size_t(k);
    int* cluster = new int[total_k_num_of_points];
    int* offset = new int[total_k_num_of_points];
    // int* index = new int[total_num_of_points * k];
    // int* graph = new int[total_num_of_points * num_of_neighbors * k];
    // float* dis = new float[total_num_of_points * num_of_neighbors * k];
    // printf("Load subgraph neighbors and distance...\n");
    // for(int i = 0; i < num_of_subgraph; i++){
    //     int num_of_subgraph_points = pre_fix_of_subgraph_size[i + 1] - pre_fix_of_subgraph_size[i];
    //     ostringstream subgraph_filename;
    //     subgraph_filename <<"subGraph"<<std::setw(4)<<std::setfill('0')<<i;
    //     string subgraph_path = graph_path + subgraph_filename.str();
    //     ifstream graph_file(subgraph_path, std::ios::binary);
    //     graph_file.read((char*)(graph + pre_fix_of_subgraph_size[i] * num_of_neighbors), num_of_subgraph_points * num_of_neighbors * sizeof(int));
    //     graph_file.close();
    //     ostringstream subgraph_dis_filename;
    //     subgraph_filename <<"subGraphDis"<<std::setw(4)<<std::setfill('0')<<i;
    //     string subgraph_dis_path = graph_path + subgraph_dis_filename.str();
    //     ifstream dis_file(subgraph_dis_path, std::ios::binary);
    //     dis_file.read((char*)(dis + pre_fix_of_subgraph_size[i] * num_of_neighbors), num_of_subgraph_points * num_of_neighbors * sizeof(float));
    //     dis_file.close();
    // }

    printf("Load cluster and offset of points...\n");
    string cluster_path = base_path + "cluster.bin";
    ifstream cluster_file(cluster_path, std::ios::binary);
    cluster_file.read((char*)cluster, total_k_num_of_points * sizeof(int));
    cluster_file.close();

    string offset_path = base_path + "clusterOffset.bin";
    ifstream offset_file(offset_path, std::ios::binary);
    offset_file.read((char*)offset, total_k_num_of_points * sizeof(int));
    offset_file.close();

    // string index_path = base_path + "new_index_of_data.bin";
    // ifstream index_file(index_path, std::ios::binary);
    // index_file.seekg(4, std::ios::cur);
    // index_file.read((char*)index, total_num_of_points * k * sizeof(int));
    // index_file.close();

    printf("Copy data to GPU...\n");
    // int* d_graph;
    // cudaMalloc(&d_graph, total_num_of_points * num_of_neighbors * k * sizeof(int));
    // cudaMemcpy(d_graph, graph, total_num_of_points * num_of_neighbors * k * sizeof(int), cudaMemcpyHostToDevice);
    // float* d_dis;
    // cudaMalloc(&d_dis, total_num_of_points * num_of_neighbors * k * sizeof(float));
    // cudaMemcpy(d_dis, dis, total_num_of_points * num_of_neighbors * k * sizeof(float), cudaMemcpyHostToDevice);
    int* d_cluster;
    cudaMalloc(&d_cluster, total_k_num_of_points * sizeof(int));
    cudaMemcpy(d_cluster, cluster, total_k_num_of_points * sizeof(int), cudaMemcpyHostToDevice);
    int* d_offset;
    cudaMalloc(&d_offset, total_k_num_of_points * sizeof(int));
    cudaMemcpy(d_offset, offset, total_k_num_of_points * sizeof(int), cudaMemcpyHostToDevice);
    // int* d_index;
    // cudaMalloc(&d_index, total_num_of_points * k * sizeof(int));
    // cudaMemcpy(d_index, index, total_num_of_points * k * sizeof(int), cudaMemcpyHostToDevice);
    int* d_pre_fix_of_subgraph_size;
    cudaMalloc(&d_pre_fix_of_subgraph_size, (num_of_subgraph + 1) * sizeof(int));
    cudaMemcpy(d_pre_fix_of_subgraph_size, pre_fix_of_subgraph_size, (num_of_subgraph + 1) * sizeof(int), cudaMemcpyHostToDevice);
    size_t* d_pre_fix_of_start_loc;
    cudaMalloc(&d_pre_fix_of_start_loc, (num_of_subgraph + 1) * sizeof(size_t));
    cudaMemcpy(d_pre_fix_of_start_loc, pre_fix_of_start_loc, (num_of_subgraph + 1) * sizeof(size_t), cudaMemcpyHostToDevice);

    printf("Start merge subgraph...\n");
    int batch_size = 10000000;
    size_t batch_elements = size_t(batch_size) * size_t(num_of_neighbors) /** size_t(k)*/;
    int* h_final_graph = new int[batch_elements];
    int* d_final_neighbor;
    cudaMalloc(&d_final_neighbor, batch_elements * sizeof(int));
    constexpr int WARP_SIZE = 32;
    constexpr int NumWarpQ = 64;
    constexpr int NumThreadQ = 1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / WARP_SIZE;
    int totalMaxBlocks = prop.multiProcessorCount * maxBlocksPerSM;
    printf("Number of blocks: %d\n", totalMaxBlocks);
    int max_iter = (total_num_of_points + batch_size - 1) / batch_size;
    string final_graph_filename = "GANNS_k" + to_string(k) + "_degree" + to_string(num_of_neighbors) + ".bin";
    ofstream final_graph(final_graph_path + final_graph_filename, std::ios::binary | std::ios::app);
    for(int iter = 0; iter < max_iter; iter++){
        int tmp_batch_size = batch_size;
        if(iter == max_iter - 1){
            tmp_batch_size = total_num_of_points - iter * batch_size;
        }
        kSubgraphMerge<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<totalMaxBlocks, 32>>>(ssd_graph, ssd_distance, d_pre_fix_of_start_loc, d_cluster, d_offset, d_final_neighbor,
                                                                                        d_pre_fix_of_subgraph_size, k, tmp_batch_size, num_of_subgraph, num_of_neighbors, total_num_of_points, iter * batch_size);
        cudaDeviceSynchronize();
        cudaMemcpy(h_final_graph, d_final_neighbor, /*size_t(k) **/ size_t(tmp_batch_size) * size_t(num_of_neighbors) * sizeof(int), cudaMemcpyDeviceToHost);
        final_graph.write((char*)h_final_graph, /*size_t(k) **/ size_t(tmp_batch_size) * size_t(num_of_neighbors) * sizeof(int));
        printf("Iteration: %d, Total merge %d points\n", iter, iter * batch_size + tmp_batch_size);
    }

    printf("Save final graph...\n");
    // cudaMemcpy(h_final_graph, d_final_neighbor, total_num_of_points * num_of_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
    // string final_graph_filename = "GANNS_k" + to_string(k) + "_degree" + to_string(num_of_neighbors) + ".bin";
    // ofstream final_graph(final_graph_path + final_graph_filename, std::ios::binary);
    // final_graph.write((char*)h_final_graph, total_num_of_points * num_of_neighbors * sizeof(int));
    final_graph.close();

}

__global__
void test(float* d_graph){
    int t_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(t_id < 64){
        printf("%d, %f\n", t_id, d_graph[796396 * 8 + t_id]);
    }
}
__global__
void changeIndex(int* d_graph, int* d_index, int num_of_points, int num_of_neighbors, int total_num_of_points){
    size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    size_t num_of_threads = size_t(blockDim.x) * size_t(gridDim.x);
    size_t num_of_elements = size_t(num_of_points) * size_t(num_of_neighbors);
    for(size_t i = 0; i < (num_of_elements + num_of_threads - 1) / num_of_threads; i++){
        size_t id = idx + i * num_of_threads;
        if(id < num_of_elements){
            int point_id = d_graph[id];
            if(point_id < num_of_points){
                d_graph[id] = d_index[point_id];
            }else{
                d_graph[id] = total_num_of_points;
            }
        }
    }
}
void subgraph::subgraphBuild(int num_of_candidates, int num_of_initial_neighbors, int num_of_points, int k){
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


    // initialize GPU page cacahe settings   
    Settings settings;
    std::vector<Controller *> ctrls(settings.n_ctrls);
    const char *const ctrls_paths[] = {"/dev/libnvm0"};
    ctrls[0] = new Controller(ctrls_paths[0], settings.nvmNamespace,
                              settings.cudaDevice, settings.queueDepth,
                              settings.numQueues);

    // initialize GPU page cacahe
    uint64_t pc_pages = settings.maxPageCacheSize / settings.pageSize;
    printf("pageSize %d, pageNum %d, cudaDevice %d\n", settings.pageSize, pc_pages, settings.cudaDevice);
    page_cache_t h_pc(settings.pageSize, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
    page_cache_d_t *d_pc = (page_cache_d_t *)(h_pc.d_pc_ptr);

    printf("There are %d subgraph\n",num_of_subgraph);
    size_t* pre_fix_of_pages = new size_t[num_of_subgraph + 1];
    pre_fix_of_pages[0] = 0;
    for(int i = 1; i <= num_of_subgraph; i++){
        pre_fix_of_pages[i] = pre_fix_of_pages[i - 1] + size_t((size_t(pre_fix_of_subgraph_size[i] - pre_fix_of_subgraph_size[i - 1]) * 
                              size_t(num_of_initial_neighbors) * sizeof(int) + settings.pageSize - 1) / settings.pageSize);
    }
    size_t distance_start_loc = pre_fix_of_pages[num_of_subgraph] * settings.pageSize + pc_pages * settings.pageSize;
    printf("Pages: %lld, dis offset:%lld \n", pre_fix_of_pages[num_of_subgraph], distance_start_loc);
    
    // string index_path = base_path + "new_index_of_data.bin";
    // ifstream index_file(index_path, std::ios::binary);
    // for(int i = 0; i < num_of_subgraph; i++){
    //     std::ostringstream subgraph_data_filename;
    //     subgraph_data_filename << "subData" << std::setw(4) << std::setfill('0') << i << ".bin";
    //     string subgraph_data_path = base_path + subgraph_data_filename.str();

    //     int num_of_subgraph_points = pre_fix_of_subgraph_size[i + 1] - pre_fix_of_subgraph_size[i];
    //     printf("Load subgraph %d data points..., %d points\n", i, num_of_subgraph_points);
    //     Data* points = new Data(subgraph_data_path, num_of_subgraph_points);

    //     GraphWrapper* graph;
    //     graph = new NavigableSmallWorldGraphWithFixedDegree(points);
    //     printf("Construct %d subgraph ...\n", i);
    //     graph->Establishment(num_of_initial_neighbors, num_of_candidates);
    //     cuda_err_chk(cudaDeviceSynchronize());
    //     ostringstream subgraph_filename;
    //     subgraph_filename <<"subGraph"<<std::setw(4)<<std::setfill('0')<<i;
    //     string subgraph_path = graph_path + subgraph_filename.str();
    //     graph->Dump(subgraph_path, num_of_initial_neighbors);
    //     int* index = new int[num_of_subgraph_points];
    //     index_file.read((char*)index, size_t(num_of_subgraph_points) * sizeof(int));
    //     int* d_index;
    //     cudaMalloc(&d_index, sizeof(int) * size_t(num_of_subgraph_points));
    //     cudaMemcpy(d_index, index, sizeof(int) * size_t(num_of_subgraph_points), cudaMemcpyHostToDevice);
    //     changeIndex<<<10000, 32>>>(graph->getDeviceGraph(), d_index, num_of_subgraph_points, num_of_initial_neighbors, num_of_points);
    //     cuda_err_chk(cudaDeviceSynchronize());
    //     delete index;
    //     cudaFree(d_index);

    //     printf("Save %d subgraph ...\n", i);
    //     // read_kernel<<<1,128>>>(graph->getDeviceGraph(), 8);
    //     // cuda_err_chk(cudaDeviceSynchronize());
    //     size_t graph_offset = pre_fix_of_pages[i] * settings.pageSize;
    //     printf("Graph:%d, Sum of pages:%lld ,offset: %lld\n", i, pre_fix_of_pages[i], graph_offset);
    //     writeArray2Disk(graph->getDeviceGraph(), size_t(num_of_subgraph_points) * size_t(num_of_initial_neighbors), 
    //                 graph_offset,
    //                 h_pc, d_pc, settings);
    //     writeArray2Disk(graph->getDeviceDistance(), size_t(num_of_subgraph_points) * size_t(num_of_initial_neighbors), 
    //                     distance_start_loc + graph_offset,
    //                     h_pc, d_pc, settings);
    //     cudaDeviceSynchronize();
    //     graph->freeSpace();
    //     delete graph;
    //     // ostringstream subgraph_filename;
    //     // subgraph_filename <<"subGraph"<<std::setw(4)<<std::setfill('0')<<i;
    //     // string subgraph_path = graph_path + subgraph_filename.str();

    //     // ostringstream subgraph_dis_filename;
    //     // subgraph_dis_filename <<"subGraphDis"<<std::setw(4)<<std::setfill('0')<<i;
    //     // string subgraph_dis_path = graph_path + subgraph_dis_filename.str();

    //     // graph->Dump(subgraph_path, subgraph_dis_path, index + pre_fix_of_subgraph_size[i], num_of_points, num_of_initial_neighbors);
    // }
    // index_file.close();
    
    uint64_t num_of_total_elements = uint64_t(pre_fix_of_pages[num_of_subgraph]) * uint64_t(settings.pageSize) / sizeof(int);
    uint64_t d_graph_total_size = uint64_t(pre_fix_of_pages[num_of_subgraph]) * uint64_t(settings.pageSize);
    uint64_t d_distance_total_size = uint64_t(pre_fix_of_pages[num_of_subgraph]) * uint64_t(settings.pageSize);

    uint64_t n_graph_pages = pre_fix_of_pages[num_of_subgraph];
    uint64_t n_distance_pages = pre_fix_of_pages[num_of_subgraph];

    uint64_t d_distance_offset = uint64_t(distance_start_loc);


    uint64_t distance_start_page = n_graph_pages + pc_pages;
    // printf("Pages: %lld, dis offset:%lld \n", n_graph_pages, d_distance_offset);

    range_t<int> *graph_range = new range_t<int>(
        (uint64_t)0, num_of_total_elements, (uint64_t)0, n_graph_pages,
        (uint64_t)0, settings.pageSize, &h_pc, settings.cudaDevice);

    range_t<float> *distance_range =
        new range_t<float>((uint64_t)0,              // start_offset
                           num_of_total_elements, // size
                           distance_start_page, // start_page
                           n_distance_pages,                        // num_pages
                           0,                 // start_buffer_page
                           settings.pageSize, // page_size
                           &h_pc, settings.cudaDevice);

    std::vector<range_t<float> *> vec_distancerange(1);
    std::vector<range_t<int> *> vec_graphrange(1);

    vec_distancerange[0] = distance_range;
    vec_graphrange[0] = graph_range;


    array_t<int> *d_graph_array = new array_t<int>(
        num_of_total_elements, 0, vec_graphrange, settings.cudaDevice);

    array_t<float> *d_distance_array =
        new array_t<float>(num_of_total_elements, d_distance_offset,
                           vec_distancerange, settings.cudaDevice);
    // // int *d_int_output;
    // // float *d_float_output;
    // // cudaMalloc((void **)&d_int_output, 128 * sizeof(int));
    // // cudaMalloc((void **)&d_float_output, 128 * sizeof(float));
    // // read_graph_kernel<<<1, 64>>>(d_graph_array->d_array_ptr, d_int_output);

    // // read_data_kernel<<<1, 64>>>(d_distance_array->d_array_ptr, d_float_output);
    printf("Start merge subgraph...\n");
    for(int i = 0; i <= num_of_subgraph; i++){
        pre_fix_of_pages[i] = pre_fix_of_pages[i] * settings.pageSize / sizeof(int);
    }
    subgraphMerge(num_of_subgraph, num_of_points, num_of_initial_neighbors, k, pre_fix_of_subgraph_size, 
                  d_graph_array->d_array_ptr, d_distance_array->d_array_ptr, pre_fix_of_pages);
}