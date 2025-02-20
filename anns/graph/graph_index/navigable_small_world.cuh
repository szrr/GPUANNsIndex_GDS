#pragma once

#include "data.h"
#include "../graph_kernel_operation/structure_on_device.cuh"
#include "nsw_graph_operations.cuh"
#include "wrapper.h"

class NavigableSmallWorldGraphWithFixedDegree : public GraphWrapper{

private:
    int num_of_initial_neighbors_;
    int num_of_maximal_neighbors_;
    int offset_shift_;
    int num_of_points_one_batch_ = 500;
    int num_of_batches_;
    float* distance_ = nullptr;
    float* d_distance = nullptr;
    idx_t* graph_ = nullptr;
    idx_t* d_graph = nullptr;
    Data* points_ = nullptr;
    val_t* d_data = nullptr;
    pair<float, int>* first_subgraph = nullptr;
    std::mt19937_64 rand_gen_ = std::mt19937_64(1234567);

    float distance(float* point_a, float* point_b) {
#if USE_L2_DIST_
        return points_->L2Distance(point_a, point_b);
#elif USE_IP_DIST_
        return points_->IPDistance(point_a, point_b);
#elif USE_COS_DIST_
        return points_->COSDistance(point_a, point_b);
#endif
    }
    void UpdateEdges(int last_point_id, int previous_point_id, float distance) {
        int position_of_neighbors_of_previous_point = previous_point_id << offset_shift_;

        int position_of_insertion = -1;

        for (int i = 0; i < num_of_maximal_neighbors_; i++) {
            if (distance < first_subgraph[position_of_neighbors_of_previous_point + i].first) {
                position_of_insertion = i;
                break;
            }
        }

        if (position_of_insertion != -1) {
            for (int i = num_of_maximal_neighbors_ - 2; i >= position_of_insertion; i--) {
                first_subgraph[position_of_neighbors_of_previous_point + i + 1] = first_subgraph[position_of_neighbors_of_previous_point + i];
            }

            first_subgraph[position_of_neighbors_of_previous_point + position_of_insertion] = std::make_pair(distance, last_point_id);
        }
    }

public:

    NavigableSmallWorldGraphWithFixedDegree(Data* data) : points_(data){
        int total_num_of_points = points_->GetNumPoints();
        num_of_batches_ = (total_num_of_points + num_of_points_one_batch_ - 1) / num_of_points_one_batch_;
        num_of_points_one_batch_ = (total_num_of_points + num_of_batches_ - 1) / num_of_batches_;
        // cudaMalloc(&d_data, sizeof(float) * points_->GetNumPoints() * points_->GetDimofPoints());
        // cudaMemcpy(d_data, points_->data_, sizeof(float) * points_->GetNumPoints() * points_->GetDimofPoints(), cudaMemcpyHostToDevice);
    }
    void freeSpace() {
        if (distance_ != nullptr) {
            delete distance_;
        }
        if (graph_ != nullptr) {
            delete graph_;
        }
        if(points_ != nullptr){
            delete points_;
        }
        if(first_subgraph != nullptr){
            delete first_subgraph;
        }
        if(d_distance != nullptr){
            cudaFree(d_distance);
        }
        if(d_graph != nullptr){
            cudaFree(d_graph);
        }
        if(d_data != nullptr){
            cudaFree(d_data);
        }
    }
    int* getDeviceGraph(){
        return d_graph;
    }

    float* getDeviceDistance(){
        return d_distance;
    }

    float* getDeviceData(){
        return d_data;
    }

    float* getHostData(){
        return points_->GetFirstPositionofPoint(0);
    }

    int getNumOfPoints(){
        return points_->GetNumPoints();
    }

    void AddPointinGraph(int point_id, float* point) {
        
        vector<pair<float, int>> neighbors;
        SearchTopK(point, (1 << offset_shift_), neighbors);

        int offset = point_id << offset_shift_;
        
        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            first_subgraph[offset + i] = neighbors[i];
        }

        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            UpdateEdges(point_id, neighbors[i].second, neighbors[i].first);
        }
    }

    void SearchTopK(float* query_point, int k, vector<pair<float, int>> &result) {
        priority_queue<pair<float, int>, vector<pair<float, int>>, std::greater<pair<float, int>>> pq;

        unordered_set<int> visited;
        int start = 0;
        visited.insert(start);
        pq.push(std::make_pair(distance(points_->GetFirstPositionofPoint(start), query_point), start));

        priority_queue<pair<float, int>> topk;
        const int max_step = 1000000;
        float min_dist = 1e100;
        
        for (int iteration_id = 0; iteration_id < max_step && !pq.empty(); iteration_id++) {
            auto now_candidate = pq.top();
			if (topk.size() == k && topk.top().first < now_candidate.first) {
                break;
            }
            
            min_dist = std::min(min_dist, now_candidate.first);
            pq.pop();
            topk.push(now_candidate);

            if (topk.size() > k) {
                topk.pop();
            }

            int offset = now_candidate.second << offset_shift_;
            
            for (int i = 0; i < num_of_maximal_neighbors_; i++) {
                int neighbor_id = first_subgraph[offset + i].second;

                if (neighbor_id >= points_->GetNumPoints()) {
                    break;
                }

                if (visited.count(neighbor_id)) {
                    continue;
                }

                pq.push(std::make_pair(distance(points_->GetFirstPositionofPoint(neighbor_id), query_point), neighbor_id));
                
                visited.insert(neighbor_id);
            }
        }
        
        result.resize(topk.size());
        int i = result.size() - 1;

        while (!topk.empty()) {
            result[i] = (topk.top());
            topk.pop();
            i--;
        }
    }

    void SearchTopKonDevice(array_t<char> *ssd_data, float* d_queries, int num_of_topk, int* &results, int num_of_query_points, int num_of_candidates, 
                            int* d_enter_cluster, GPUIndex* d_rvq_index, Timer* &graphSearch, int search_width, int num_of_warmup_vectors, float *d_warmup_vectors){

        int num_of_topk_ =  num_of_topk;//pow(2.0, ceil(log(num_of_topk) / log(2)));
        cudaMallocHost(&results, sizeof(int) * num_of_query_points * num_of_topk);
        // results = new int[num_of_query_points * num_of_topk];
        int num_of_explored_points = num_of_candidates;
        num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));
        DisplaySearchParameters(num_of_topk, num_of_explored_points);
        //enterPointsSearchTopK(queries, num_of_query_points, 128, enterPoints);
        CUDA_SYNC_CHECK();
    	NSWGraphOperations::Search(ssd_data, d_data, d_queries, graph_, results, num_of_query_points, points_->GetNumPoints(), 
                                   points_->GetDimofPoints(), offset_shift_, num_of_topk_, num_of_candidates, num_of_explored_points, 
                                   d_enter_cluster, d_rvq_index, graphSearch, search_width, num_of_warmup_vectors, d_warmup_vectors);

    }

    void Establishment(int num_of_initial_neighbors, int num_of_candidates){
        float* d_points;
        KernelPair<float, int>* d_neighbors;
        KernelPair<float, int>* d_neighbors_backup;

        num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));
        num_of_initial_neighbors_ = pow(2.0, ceil(log(num_of_initial_neighbors) / log(2)));

        offset_shift_ = log(num_of_initial_neighbors_) / log(2) + 1;
        num_of_maximal_neighbors_ = (1 << offset_shift_);

        size_t num_of_data_elements = size_t(points_->GetNumPoints()) * size_t(points_->GetDimofPoints());
        error_check(cudaMalloc(&d_points, sizeof(float) * num_of_data_elements), __LINE__);
        error_check(cudaGetLastError(), __LINE__);
        error_check(cudaMemcpy(d_points, points_->GetFirstPositionofPoint(0), sizeof(float) * num_of_data_elements, cudaMemcpyHostToDevice), __LINE__);

        error_check(cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * (size_t(points_->GetNumPoints()) << offset_shift_)), __LINE__); 
        error_check(cudaGetLastError(), __LINE__);
        // error_check(cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (size_t(points_->GetNumPoints()) << offset_shift_)), __LINE__);
        
        DisplayGraphParameters(num_of_candidates);

        pair<float, int> neighbor_intialisation = std::make_pair(MAX, points_->GetNumPoints());
        vector<pair<float, int>> substitute(size_t(num_of_points_one_batch_) * size_t(num_of_maximal_neighbors_), neighbor_intialisation);

        first_subgraph = new pair<float, int>[size_t(num_of_points_one_batch_) * size_t(num_of_maximal_neighbors_)];

        std::copy(substitute.begin(), substitute.end(), first_subgraph);
        
        for (int i = 1; i < num_of_points_one_batch_; i++) {
            AddPointinGraph(i, points_->GetFirstPositionofPoint(i));
        }
        // cudaMallocHost(&graph_, sizeof(int) * (size_t(points_->GetNumPoints()) << offset_shift_));
        // distance_ = new float[size_t(points_->GetNumPoints()) * size_t(num_of_initial_neighbors)];

        // cudaMalloc(&d_graph, sizeof(int) * (size_t(points_->GetNumPoints()) * size_t(num_of_initial_neighbors)));

        // cudaMalloc(&d_distance, sizeof(float) * (size_t(points_->GetNumPoints()) * size_t(num_of_initial_neighbors)));
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        NSWGraphOperations::LocalGraphConstructionBruteForce(points_->GetFirstPositionofPoint(0), offset_shift_, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_, num_of_batches_, num_of_points_one_batch_,
                                                                d_points, d_neighbors, d_neighbors_backup);
        error_check(cudaGetLastError(), __LINE__);
        NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(d_points, graph_, d_graph,points_->GetNumPoints(), points_->GetDimofPoints(), offset_shift_, num_of_initial_neighbors_, num_of_batches_, 
                                                                    num_of_points_one_batch_, d_neighbors, d_neighbors_backup, num_of_maximal_neighbors_, num_of_candidates, first_subgraph, distance_, d_distance);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        cout << "Running time: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000 << " seconds" << endl;
        // cudaFree(d_neighbors);
        // cudaFree(d_neighbors_backup);
        // cudaFree(d_points);
        substitute.clear(); 
        substitute.shrink_to_fit();
    }

    

    void Dump(string graph_name, string dis_name, int* index_of_points, int total_num_of_points, int num_of_neighbors){
        for(int i = 0; i < points_->GetNumPoints(); i++){
            idx_t *loc = graph_ + i * num_of_neighbors;
            for(int l = 0; l < num_of_neighbors; l++){
                if(loc[l] < points_->GetNumPoints()){
                    loc[l] = index_of_points[loc[l]];
                }else{
                    loc[l] = total_num_of_points;
                }
            }
        }
        ofstream out_graph(graph_name, std::ios::binary);
        out_graph.write((char*)graph_, sizeof(int) * (points_->GetNumPoints() * num_of_neighbors));
        out_graph.close();

        ofstream out_dis(dis_name, std::ios::binary);
        out_dis.write((char*)distance_, sizeof(float) * (points_->GetNumPoints() * num_of_neighbors));
        out_dis.close();

        // ofstream out_final_graph(final_graph_name, std::ios::binary);
        // out_final_graph.write((char*)graph_, sizeof(int) * (points_->GetNumPoints() << offset_shift_));
        // out_final_graph.close();
    }

    void Dump(string graph_name, int num_of_neighbors){
        ofstream out_descriptor(graph_name, std::ios::binary);
        
        out_descriptor.write((char*)graph_, sizeof(int) * points_->GetNumPoints() * num_of_neighbors);
        out_descriptor.close();
    }

    void Load(string graph_path, int degree_of_graph){
        int num_of_points = points_->GetNumPoints();
        offset_shift_ = ceil(log(degree_of_graph) / log(2));
        // cudaMallocHost(&graph_, sizeof(int) * (num_of_points << offset_shift_));
        // ifstream in_descriptor(graph_path, std::ios::binary);
        // if (!in_descriptor.is_open()) {
        //     exit(1);
        // }
        // in_descriptor.seekg(790, std::ios::beg);
        // in_descriptor.read((char*)(graph_), (num_of_points << offset_shift_) * sizeof(int));
        // ifstream in_descriptor(graph_path, std::ios::binary);
        // int num_of_points = points_->GetNumPoints();

        // in_descriptor.seekg(0, std::ios::end);
        // long long file_size = in_descriptor.tellg();
        // offset_shift_ = file_size / num_of_points / sizeof(int);
        // offset_shift_ = ceil(log(offset_shift_) / log(2));
        
        // in_descriptor.seekg(0, std::ios::beg);
        // cudaMallocHost(&graph_, sizeof(int) * (num_of_points << offset_shift_));
        // in_descriptor.read((char*)graph_, sizeof(int) * (num_of_points << offset_shift_));
        // in_descriptor.close();
    }

    void DisplayGraphParameters(int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           d_min = " << num_of_initial_neighbors_ << endl;
        cout << "           d_max = " << num_of_maximal_neighbors_ << endl;
        cout << "           l_n = " << num_of_candidates << endl << endl;
    }

    void DisplaySearchParameters(int num_of_topk, int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           the number of topk = " << num_of_topk << endl;
        cout << "           e = " << num_of_candidates << endl << endl;
    }
};

