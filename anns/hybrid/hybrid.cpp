/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#include <iostream>
#include <vector>
#include "hybrid.h"

void testCopyIndexToGPU(int points_num, int numCoarseCentroids, int numFineCentroids, GPUIndex* d_index) {

    srand((unsigned int)(time(0)));
    vector<std::vector<std::vector<idx_t>>> index;
    index.resize(numCoarseCentroids);
    for (int i = 0; i < numCoarseCentroids; ++i) {
        index[i].resize(numFineCentroids);
        for (int j = 0; j < numFineCentroids; ++j) {
            index[i][j].resize((rand()%200)+1);
            if(i == 0 && j == 0){
                index[i][j].resize(0);
            }
            for(int l = 0; l < index[i][j].size(); l++){
                index[i][j][l] = rand()%points_num;
            }
        }
    }
    d_index->numCoarseCentroids = numCoarseCentroids;
    d_index->numFineCentroids = numFineCentroids;

    // 分配指针数组
    int** hostIndices = new int*[numCoarseCentroids * numFineCentroids];
    int* hostSizes = new int[numCoarseCentroids * numFineCentroids];

    // 分配数据并拷贝到GPU
    for (int i = 0; i < numCoarseCentroids; ++i) {
        for (int j = 0; j < numFineCentroids; ++j) {
            int idx = i * numFineCentroids + j;
            hostSizes[idx] = index[i][j].size();
            if (hostSizes[idx] > 0) {
                CUDA_CHECK(cudaMalloc(&hostIndices[idx], hostSizes[idx] * sizeof(idx_t)));
                CUDA_CHECK(cudaMemcpy(hostIndices[idx], index[i][j].data(), hostSizes[idx] * sizeof(idx_t), cudaMemcpyHostToDevice));
            } else {
                hostIndices[idx] = nullptr;
            }
        }
    }

    // 分配GPU端指针
    int** deviceIndices;
    CUDA_CHECK(cudaMalloc(&deviceIndices, numCoarseCentroids * numFineCentroids * sizeof(int*)));
    int* deviceSizes;
    CUDA_CHECK(cudaMalloc(&deviceSizes, numCoarseCentroids * numFineCentroids * sizeof(int)));

    // 拷贝指针数组到GPU
    CUDA_CHECK(cudaMemcpy(deviceIndices, hostIndices, numCoarseCentroids * numFineCentroids * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSizes, hostSizes, numCoarseCentroids * numFineCentroids * sizeof(int), cudaMemcpyHostToDevice));

    // 设置 GPUIndex 的成员
    d_index->indices = deviceIndices;
    d_index->sizes = deviceSizes;

    // 释放临时数组
    delete[] hostIndices;
    delete[] hostSizes;
}

void testHybridSearch(int* d_enter_cluster, int num_queries){
    int* h_enter_cluster = new int[num_queries];
    //cudaMemcpy(h_enter_cluster, d_enter_cluster, sizeof(int) * num_queries, cudaMemcpyDeviceToHost);
    srand((unsigned int)(time(0)));
    for(int i=0; i<num_queries; i++){
        //printf("%d ",h_enter_cluster[i]);
        h_enter_cluster[i] = rand() % 10000;
    }
    cudaMemcpy(d_enter_cluster, h_enter_cluster, sizeof(int)*num_queries, cudaMemcpyHostToDevice);
}



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
    int* d_enter_cluster;
    cudaMalloc((void**)&d_enter_cluster, num_queries * sizeof(int));
    graphSearch[0].Stop();
    

    GPUIndex* d_rvq_index = rvq->get_gpu_index();
    //test
    //testCopyIndexToGPU(1000000, 100, 100, d_rvq_index);
    //testHybridSearch(d_enter_cluster, num_queries);

    Timer rvqSearch;
    rvq->search(d_queries, num_queries, d_enter_cluster);
    rvqSearch.Start();
    rvq->search(d_queries, num_queries, d_enter_cluster);
    rvqSearch.Stop();

    // int* h_enter_cluster;
    // h_enter_cluster = new int[20000];
    // cudaMemcpy(h_enter_cluster, d_enter_cluster, sizeof(int) * num_queries, cudaMemcpyDeviceToHost);
    //enterClusterSave(num_queries, h_enter_cluster, "/home/ErHa/GANNS_Res/h_enter_cluster.bin");
    // enterClusterLoad(num_queries, h_enter_cluster, "/home/ErHa/GANNS_Res/h_enter_cluster.bin");
    // cudaMemcpy(d_enter_cluster, h_enter_cluster, sizeof(int) * num_queries, cudaMemcpyHostToDevice);

    // for(int i=0; i<num_queries; i++){
    //     printf("%d,%d  ",i,h_enter_cluster[i]);
    // }

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