/**
 * @author szr
 * @date 2024/6/17
 * @brief hybrid index with two-layer RVQ and one layer anns graph
 * 
 * **/

#include <iostream>
#include <vector>
#include "../common.h"
#include "hybrid.h"

void hybrid::train(float* trainVectorData, num_t numTrainVectors){
    rvq->train(trainVectorData, numTrainVectors);
}

void hybrid::build(float* buildVectorData, num_t numVectors){
    rvq->build(buildVectorData, numVectors);
    graph->build(buildVectorData, numVectors); // add graph build
}

void hybrid::search(float* queries, int numQueries, std::vector<std::vector<idx_t>>& res){
    std::vector<std::vector<idx_t>> enterPoints;
    rvq->search(queries, numQueries, enterPoints);
    graph->search(queries, numQueries, enterPoints, res); // add graph build
}