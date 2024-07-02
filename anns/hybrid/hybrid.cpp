/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#include <iostream>
#include <vector>
#include "hybrid.h"

void hybrid::hybrid_train(float* trainVectorData, num_t numTrainVectors){
    // for(int k=0;k<10000;k++){
    //     cout<<k<<" "<<endl;
    //     for(int i=0;i<128;i++){
    //         cout<<trainVectorData[k*128 + i]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    rvq->train(trainVectorData, numTrainVectors);
}

void hybrid::hybrid_build(float* buildVectorData, num_t numVectors){
    rvq->build(buildVectorData, numVectors);
    // std::vector<std::vector<std::vector<idx_t>>> rvqIndex = rvq->get_index(); // get rvq two layer index
    // graph->build(buildVectorData, numVectors, rvqIndex); // add graph build
}

void hybrid::hybrid_search(float* queries, int topk, int* &results, int num_queries, int num_candidates){
    float *d_queries;
    CUDA_CHECK(cudaMalloc((void **)&d_queries, sizeof(float) * num_queries * dim_));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, sizeof(float) * num_queries * dim_, cudaMemcpyHostToDevice));
    
    // Todo: enter points should be cpu or gpu?
    int* d_enter_cluster;
    cudaMalloc((void**)&d_enter_cluster, numQueries * sizeof(int));
    Timer rvqSearch;
    for(int i=0; i<2; i++){
        rvqSearch.Start();
        rvq->search(d_queries, num_queries, d_enter_cluster);
        rvqSearch.Stop();
    }
    // Todo: graph input: cluster_id and d_index_

    Timer* graphSearch = new Timer[4];
    graph->SearchTopKonDevice(queries, topk, results, num_queries, num_candidates, enterPoints, graphSearch); // add graph build
    std::cout<<"Find enter points time: "<<rvqSearch.DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Transfer data time: "<<graphSearch[0].DurationInMilliseconds() + graphSearch[3].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Transfer enter points time: "<<graphSearch[1].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"Search time: "<<graphSearch[2].DurationInMilliseconds()<<" ms"<<std::endl;
    std::cout<<"QPS without transfer enter points: "<<int64_t(double(num_queries)/((rvqSearch.DurationInMilliseconds()+graphSearch[2].DurationInMilliseconds()+graphSearch[3].DurationInMilliseconds()+graphSearch[0].DurationInMilliseconds())/1000))<<std::endl;
    std::cout<<"QPS: "<<int64_t(double(num_queries)/((rvqSearch.DurationInMilliseconds()+graphSearch[0].DurationInMilliseconds()+graphSearch[1].DurationInMilliseconds() + graphSearch[2].DurationInMilliseconds())/1000))<<std::endl;
    
}

void hybrid::hybrid_save(const std::string& filename) {
    rvq->save(filename);
}

void hybrid::hybrid_load(const std::string& filename) {
    rvq->load(filename);
    // todo: load graph index?
}