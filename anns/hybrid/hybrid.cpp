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

void hybrid::hybrid_build(const std::string& filename, int num_of_points){
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }
    // in.seekg(4, std::ios::cur);
    vector<vector<int>> num_of_subgraph_points;
    num_of_subgraph_points.resize(rvq->get_numCoarseCentroid());
    for (auto& inner : num_of_subgraph_points){
        inner.resize(rvq->get_numFineCentroid());
    }

    vector<vector<int>> new_index_of_points;
    new_index_of_points.resize(rvq->get_numFineCentroid() * rvq->get_numCoarseCentroid());
    int num_of_batch_points = 1000000;
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
        // char* sub_data_char = new char[tmp_num_of_batch_points * dim_];
        // for (int i = 0; i < tmp_num_of_batch_points; i++) {
        //     in.seekg(4, std::ios::cur);
        //     in.read((char*)(sub_data_char + i * dim_), dim_);
        //     for(int l = 0;l < dim_; l++){
        //         sub_data[i * dim_ + l] = float(sub_data_char[i * dim_ + l]);
        //     }
        // }
        float* d_sub_data;
        CUDA_CHECK(cudaMalloc((void **)&d_sub_data, sizeof(float) * tmp_num_of_batch_points * dim_));
        CUDA_CHECK(cudaMemcpy(d_sub_data, sub_data, sizeof(float) * tmp_num_of_batch_points * dim_, cudaMemcpyHostToDevice));
        //build
        rvq->build(d_sub_data, tmp_num_of_batch_points, num_of_batch_points * iter);
        printf("Save subgraph data\n");
        rvq->saveSubgraphData(sub_data, num_of_subgraph_points, num_of_batch_points, iter);
        cudaFree(d_sub_data);
        delete[] sub_data;
        // delete[] sub_data_char;
    }
    in.close();
    rvq->saveSubgraphIndex(num_of_points);
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

void hybrid::hybrid_search(float* queries, int topk, int* &results, int num_queries, int num_candidates, int search_width){

    Timer* graphSearch = new Timer[4];
    cudaDeviceSynchronize();
    graphSearch[0].Start();
    float *d_queries;
    cudaMalloc((void **)&d_queries, sizeof(float) * num_queries * dim_);
    cudaMemcpy(d_queries, queries, sizeof(float) * num_queries * dim_, cudaMemcpyHostToDevice);
    graphSearch[0].Stop();
    

    GPUIndex* d_rvq_index = rvq->get_gpu_index();

    Timer rvqSearch;
    // int* test;
    // cudaMalloc((void**)&test, num_queries * sizeof(int));
    // rvq->search(d_queries, num_queries, test);
    rvqSearch.Start();
    int* d_enter_cluster;
    // cudaMalloc((void**)&d_enter_cluster, num_queries * sizeof(int));
    // rvq->search(d_queries, num_queries, d_enter_cluster);
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


    graph->SearchTopKonDevice(d_queries, topk, results, num_queries, num_candidates, d_enter_cluster, d_rvq_index, graphSearch, search_width); // add graph build
    std::cout<<"Find enter points time: "<<rvqSearch.DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Transfer data time: "<<graphSearch[0].DurationInMilliseconds() + graphSearch[1].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds()<<" ms"<<std::endl;
    //std::cout<<"Transfer enter points time: "<<graphSearch[1].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Search time: "<<graphSearch[2].DurationInMilliseconds()<<" ms"<<std::endl;
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