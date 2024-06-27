/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#pragma once

#include "../common.h"
#include "../RVQ/RVQ.h"
#include "../graph/graph_index/navigable_small_world.cuh"
#include "../graph/graph_index/data.h"


class hybrid {
public:
    hybrid(int dim, Data* data, string graph_path, int numCoarseCentroids = 100, int numFineCentroids = 100, int ef = 16, int efConstruction = 64) {
        rvq = new RVQ(dim, numCoarseCentroids, numFineCentroids);
        graph = new NavigableSmallWorldGraphWithFixedDegree(data);
        graph->Load(graph_path);
    }

    ~hybrid() {
        delete rvq;
        delete graph;
    }

    void hybrid_train(float* trainVectorData, num_t numTrainVectors);
    void hybrid_build(float* buildVectorData, num_t numVectors);
    void hybrid_search(float* queries, int num_of_topk, int* &results, int num_of_query_points, int num_of_candidates);

    void hybrid_save(const std::string& filename);
    void hybrid_load(const std::string& filename);

//private:
    RVQ* rvq;
    GraphWrapper* graph;
};

