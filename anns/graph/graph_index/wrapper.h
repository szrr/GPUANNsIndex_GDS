#pragma once
#include "../../RVQ/RVQ.cuh"
using namespace std;

class GraphWrapper{

public:
	virtual void Dump(string graph_name, string dis_name, int* index_of_points, int total_num_of_points, int num_of_neighbors) = 0;
	virtual void Dump(string graph_name, int num_of_neighbors) = 0;
	virtual void Establishment(int num_of_initial_neighbors, int num_of_candidates) = 0;
	virtual void Load(string graph_path, int degree_of_graph) = 0;
	virtual void SearchTopKonDevice(array_t<char> *ssd_data, float* queries, int num_of_topk, int* &results, int num_of_query_points, 
									int num_of_candidates,int* d_enter_cluster, GPUIndex* d_rvq_index,Timer* &graphSearch, int search_width,
									int num_of_warmup_vectors, float *d_warmup_vectors) = 0;
	virtual void DisplayGraphParameters(int num_of_candidates) = 0;
	virtual void DisplaySearchParameters(int num_of_topk, int num_of_candidates) = 0;
	virtual int* getDeviceGraph() = 0;
    virtual float* getDeviceDistance() = 0;
	virtual float* getDeviceData() = 0;
	virtual float* getHostData() = 0;
    virtual int getNumOfPoints() = 0;
	virtual void freeSpace() = 0;

private:
	
};