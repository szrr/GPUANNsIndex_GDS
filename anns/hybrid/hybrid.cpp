/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#include <iostream>
#include <vector>
#include "hybrid.h"

void hybrid::hybrid_train(){
    rvq->train(graph->getHostData(), graph->getNumOfPoints());
}

void hybrid::hybrid_build(){
    rvq->build(graph->getDeviceData(), graph->getNumOfPoints());
    // std::vector<std::vector<std::vector<idx_t>>> rvqIndex = rvq->get_index(); // get rvq two layer index
    // graph->build(buildVectorData, numVectors, rvqIndex); // add graph build
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

void hybrid::hybrid_search(float* queries, int topk, int* &results, int num_queries, int num_candidates){

    Timer* graphSearch = new Timer[4];
    graphSearch[0].Start();
    float *d_queries;
    CUDA_CHECK(cudaMalloc((void **)&d_queries, sizeof(float) * num_queries * dim_));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, sizeof(float) * num_queries * dim_, cudaMemcpyHostToDevice));
    graphSearch[0].Stop();
    

    GPUIndex* d_rvq_index = rvq->get_gpu_index();

    Timer rvqSearch;
    int* test;
    cudaMalloc((void**)&test, num_queries * sizeof(int));
    rvq->search(d_queries, num_queries, test);
    rvqSearch.Start();
    int* d_enter_cluster;
    cudaMalloc((void**)&d_enter_cluster, num_queries * sizeof(int));
    rvq->search(d_queries, num_queries, d_enter_cluster);
    rvqSearch.Stop();

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


    graph->SearchTopKonDevice(d_queries, topk, results, num_queries, num_candidates, d_enter_cluster, d_rvq_index, graphSearch); // add graph build
    std::cout<<"Find enter points time: "<<rvqSearch.DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Transfer data time: "<<graphSearch[0].DurationInMilliseconds() + graphSearch[1].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds()<<" ms"<<std::endl;
    //std::cout<<"Transfer enter points time: "<<graphSearch[1].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Search time: "<<graphSearch[2].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"QPS without RVQ: "<<int64_t(double(num_queries)/((graphSearch[0].DurationInMilliseconds() + graphSearch[1].DurationInMilliseconds()
    + graphSearch[2].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds())/1000))<<std::endl;
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