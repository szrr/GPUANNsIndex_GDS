/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#pragma once
#include <iostream>
#include <vector>
#include "../common.h"
#include "../RVQ/RVQ.cuh"
#include "../graph/graph_index/navigable_small_world.cuh"
#include "../graph/graph_index/data.h"



class hybrid {
public:
    hybrid(int dim, Data* data, string graph_path, int numCoarseCentroids = 100, int numFineCentroids = 100, int ef = 16, int efConstruction = 64) {
        dim_ = dim;
        rvq = new RVQ(dim, numCoarseCentroids, numFineCentroids);
        graph = new NavigableSmallWorldGraphWithFixedDegree(data);
        graph->Load(graph_path, ef);
    }

    ~hybrid() {
        delete rvq;
        delete graph;
    }

    void hybrid_train(string base_path, int num_of_points);
    void hybrid_build(const std::string& filename, int num_of_points);
    void hybrid_search(array_t<char> *ssd_data, float* queries, int num_of_topk, int* &results, 
                       int num_of_query_points, int num_of_candidates, int search_width);

    void hybrid_save(const std::string& filename);
    void hybrid_load(const std::string& filename);
    void hybrid_loadCodebook(const std::string& filename);

//private:
    RVQ* rvq;
    GraphWrapper* graph;
    int dim_;
};

/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

void hybrid::hybrid_train(string base_path, int num_of_points){
    rvq->train(base_path, num_of_points);
}

void hybrid::hybrid_build(const std::string& filename, int num_of_points){
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }
    in.seekg(8, std::ios::cur);
    vector<vector<int>> num_of_subgraph_points;
    num_of_subgraph_points.resize(rvq->get_numCoarseCentroid());
    for (auto& inner : num_of_subgraph_points){
        inner.resize(rvq->get_numFineCentroid());
    }

    vector<vector<int>> new_index_of_points;
    new_index_of_points.resize(rvq->get_numFineCentroid() * rvq->get_numCoarseCentroid());
    int num_of_batch_points = 10000000;
    if(num_of_batch_points > num_of_points) num_of_batch_points = num_of_points;
    int iteration = ceil(num_of_points / num_of_batch_points);
    printf("%d points.Batch size: %d\n",num_of_points, num_of_batch_points);
    for(int iter = 0; iter < iteration; iter++){
        int tmp_num_of_batch_points = num_of_batch_points;
        printf("Iteration: %d, Add %d points in RVQ\n",iter, tmp_num_of_batch_points);
        if(iter == iteration-1) tmp_num_of_batch_points = num_of_points - num_of_batch_points * iter;
        //读取数据
        float* sub_data = new float[tmp_num_of_batch_points * dim_];
        in.read((char*)sub_data, dim_ * tmp_num_of_batch_points * sizeof(float));
        float* d_sub_data;
        CUDA_CHECK(cudaMalloc((void **)&d_sub_data, sizeof(float) * tmp_num_of_batch_points * dim_));
        CUDA_CHECK(cudaMemcpy(d_sub_data, sub_data, sizeof(float) * tmp_num_of_batch_points * dim_, cudaMemcpyHostToDevice));
        //build
        rvq->build(d_sub_data, tmp_num_of_batch_points, num_of_batch_points * iter);
        printf("Save subgraph data\n");
        // rvq->saveSubgraphData(sub_data, num_of_subgraph_points, num_of_batch_points, iter);
        cudaFree(d_sub_data);
        delete[] sub_data;
        // delete[] sub_data_char;
    }
    in.close();
    // rvq->saveSubgraphIndex(num_of_points);
}

void enterClusterSave(int num_queries, int* h_enter_cluster, const string& filename){
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving." << std::endl;
        return;
    }
    out.write(reinterpret_cast<char*>(h_enter_cluster), sizeof(h_enter_cluster) * num_queries);
    out.close();
}

void enterClusterLoad(int num_queries, int* &h_enter_cluster, const string& filename){
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }
    in.read(reinterpret_cast<char*>(h_enter_cluster), sizeof(h_enter_cluster) * num_queries);
    in.close();
}
__global__ void map2starling(int* starling_index, int** d_rvq_indices, int* d_rvq_indices_size){
    int t_id = threadIdx.x;
    int query_id = blockIdx.x;
    int num_of_threads = blockDim.x;
    int num_of_enter_points = d_rvq_indices_size[query_id];
    int* enter_points_pos = d_rvq_indices[query_id];
    for(int i = 0; i < (num_of_enter_points + num_of_threads - 1)/ num_of_threads; i++){
        int idx = t_id + i * num_of_threads;
        if(idx < num_of_enter_points){
            enter_points_pos[idx] = starling_index[enter_points_pos[idx]];
        }
    }
}
void hybrid::hybrid_search(array_t<char> *ssd_data, float* queries, int topk, int* &results, int num_queries, int num_candidates, int search_width){

    Timer* graphSearch = new Timer[4];
    graphSearch[0].Start();
    float *d_queries;
    cudaMalloc((void **)&d_queries, sizeof(float) * num_queries * dim_);
    cudaMemcpy(d_queries, queries, sizeof(float) * num_queries * dim_, cudaMemcpyHostToDevice);
    graphSearch[0].Stop();
    CUDA_SYNC_CHECK();


    Timer rvqSearch;
    // int* test;
    // cudaMalloc((void**)&test, num_queries * sizeof(int));
    // rvq->search(d_queries, num_queries, test);
    // cudaFree(test);
    rvqSearch.Start();
    int* d_enter_cluster;
    cudaMalloc((void**)&d_enter_cluster, num_queries * sizeof(int));
    rvq->search(d_queries, num_queries, d_enter_cluster);
    int* h_enter_cluster = new int[num_queries];
    cudaMemcpy(h_enter_cluster, d_enter_cluster, sizeof(int) * num_queries, cudaMemcpyDeviceToHost);
    GPUIndex* d_rvq_index = new GPUIndex;
    // rvq->copyIndexToGPU(d_rvq_index, num_queries, h_enter_cluster, dim_);
    rvqSearch.Stop();
    rvq->copyIndexToGPU(d_rvq_index);
    int num_of_warmup_vectors = 0;
    float *d_warmup_vectors;
    rvq->get_warmup_vectors(d_warmup_vectors, num_of_warmup_vectors);
    // std::ifstream in("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/starlingIndex.bin", std::ios::binary);
    // int* starling_index = new int[100000000];
    // in.read((char*)starling_index, size_t(100000000) * sizeof(int));
    // in.close();
    // int* d_starling_index;
    // cudaMalloc(&d_starling_index, size_t(100000000) * sizeof(int));
    // cudaMemcpy(d_starling_index, starling_index, size_t(100000000) * sizeof(int), cudaMemcpyHostToDevice);
    // map2starling<<<num_queries, 64>>>(d_starling_index, d_rvq_index->indices, d_rvq_index->sizes);
    // cudaFree(d_starling_index);
    // delete[] starling_index;
    printf("RVQ done...\n");
    // delete[] h_enter_cluster;

    // int* h_enter_cluster;
    // h_enter_cluster = new int[20000];
    // cudaMemcpy(h_enter_cluster, d_enter_cluster, sizeof(int) * num_queries, cudaMemcpyDeviceToHost);
    //enterClusterSave(num_queries, h_enter_cluster, "/home/ErHa/GANNS_Res/h_enter_cluster.bin");
    // enterClusterLoad(num_queries, h_enter_cluster, "/home/ErHa/GANNS_Res/h_enter_cluster.bin");
    // cudaMemcpy(d_enter_cluster, h_enter_cluster, sizeof(int) * num_queries, cudaMemcpyHostToDevice);


    // Todo: graph input
    // gpu query vectors: (float*) d_queries
    // gpu search result cluster id: (int*) d_enter_cluster
    // gpu rvq index: GPUIndex* d_rvq_index = rvq->get_gpu_index()
    // gpu cluster size: int size = d_rvq_index->size[cluster_id]
    // gpu cluster points id: int* point_id = d_rvq_index->indices[cluster_id]


    graph->SearchTopKonDevice(ssd_data, d_queries, topk, results, num_queries, num_candidates, d_enter_cluster, d_rvq_index, graphSearch, search_width, 
                              num_of_warmup_vectors, d_warmup_vectors); // add graph build
    std::cout<<"Find enter points time: "<<rvqSearch.DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Transfer data time: "<<graphSearch[0].DurationInMilliseconds() + graphSearch[1].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds()<<" ms"<<std::endl;
    //std::cout<<"Transfer enter points time: "<<graphSearch[1].DurationInMilliseconds()<<" ms"<<std::endl;
    // std::cout<<"Search time: "<<graphSearch[2].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"QPS without RVQ: "<<int64_t(double(num_queries)/((graphSearch[0].DurationInMilliseconds() + graphSearch[1].DurationInMilliseconds()
    + graphSearch[2].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds())/1000))<<std::endl;
    std::cout<<"QPS without Transfer: "<<int64_t(double(num_queries)/((rvqSearch.DurationInMilliseconds() + graphSearch[2].DurationInMilliseconds())/1000))<<std::endl;
    std::cout<<"QPS: "<<int64_t(double(num_queries)/((rvqSearch.DurationInMilliseconds()+graphSearch[0].DurationInMilliseconds() + 
    graphSearch[1].DurationInMilliseconds() + graphSearch[2].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds())/1000))<<std::endl;

    cudaFree(d_rvq_index);
	cudaFree(d_enter_cluster);
	cudaFree(d_queries);
}

void hybrid::hybrid_save(const std::string& filename) {
    rvq->save(filename);
}

void hybrid::hybrid_load(const std::string& filename) {
    rvq->load(filename);
    // todo: load graph index?
}

void hybrid::hybrid_loadCodebook(const std::string& filename){
    rvq->loadCodebook(filename);
}