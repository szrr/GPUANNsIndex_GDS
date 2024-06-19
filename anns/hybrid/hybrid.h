/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#pragma once

#include "../RVQ/RVQ.h"

class hybrid {
public:
    hybrid(int dim, int numCoarseCentroids = 100, int numFineCentroids = 100, int ef = 16, int efConstruction = 64, int klist = 100) {
        rvq = new RVQ(dim, numCoarseCentroids, numFineCentroids);
        graph = new GraphIndex(ef, efConstruction, klist);
    }

    ~hybrid() {
        delete rvq;
        delete graph;
    }

    void hybrid::train(float* trainVectorData, idx_t numTrainVectors);
    void hybrid::build(float* buildVectorData, num_t numVectors);
    void hybrid::search(float* queries, int numQueries, std::vector<std::vector<idx_t>>& res);

private:
    RVQ* rvq;
    GraphIndex* graph;
};