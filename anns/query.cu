#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "hybrid/hybrid.h"

using namespace std;

void LoadGroundtruth(int* &groundtruth, string groundtruth_path, int &k_of_groundtruth){

    ifstream in_descriptor(groundtruth_path, std::ios::binary);
    if (!in_descriptor.is_open()) {
        cout << "the file path is wrong." << endl;
        exit(-1);
    }

    in_descriptor.read((char*)&k_of_groundtruth, 4);

    in_descriptor.seekg(0, std::ios::end);
    long long file_size = in_descriptor.tellg();
    int num_of_points = file_size / (k_of_groundtruth + 1) / 4;
    cout<<"k_of_groundtruth: "<<k_of_groundtruth<<" query_num: "<<num_of_points<<endl;
    cudaMallocHost(&groundtruth, num_of_points * k_of_groundtruth * sizeof(int));

    in_descriptor.seekg(0, std::ios::beg);

    for (int i = 0; i < num_of_points; i++) {
        in_descriptor.seekg(4, std::ios::cur);
        in_descriptor.read((char*)(groundtruth + i * k_of_groundtruth), k_of_groundtruth * sizeof(int));
    }

    in_descriptor.close();

}

void ComputeRecall(int* results, int* groundtruth, int num_of_queries, int num_of_topk, int k_of_groundtruth, int num_of_topk_, float &recall){
    int num_of_right_candidates = 0;

    for (int i = 0; i < num_of_queries; i++) {
        for (int j = 0; j < num_of_topk; j++) {
            int crt_candidate_id = results[i * num_of_topk_ + j];
            //cout<<"1"<<endl;
            int* position_of_candidate = NULL;
            position_of_candidate = find(groundtruth + i * k_of_groundtruth, groundtruth + i * k_of_groundtruth + num_of_topk, crt_candidate_id);
            //cout<<"2"<<endl;
            if (position_of_candidate != groundtruth + i * k_of_groundtruth + num_of_topk) {
                num_of_right_candidates++;
            }
        }
        /*if(i == 0){
            for (int j = 0; j < num_of_topk; j++) {
                int crt_candidate_id = results[i * num_of_topk_ + j];
                cout<<crt_candidate_id<<" ";
            }
            cout<<endl;
            for (int j = 0; j < num_of_topk; j++) {
                cout<<groundtruth[i * k_of_groundtruth + j]<<" ";
            }
             cout<<endl;
        }*/
    }

    recall = (float)num_of_right_candidates / (num_of_queries * num_of_topk);
}

int main(int argc,char** argv){

    //required variables from external input
    string base_path = argv[1];
    string query_path = argv[2];
    string graph_path = argv[3];
    string groundtruth_path = argv[4];
    int num_of_candidates = atoi(argv[5]);
    int num_of_topk = atoi(argv[6]);
    int num_of_points = atoi(argv[7]);

    cout << "Load groundtruth..." << endl << endl;
    int* groundtruth = NULL;
    int k_of_groundtruth;
    LoadGroundtruth(groundtruth, groundtruth_path, k_of_groundtruth);

    cout << "Load data points and query points..." << endl << endl;
    Data* points = new Data(base_path,num_of_points);
    Data* query_points = new Data(query_path);
    
    cout << "Load proximity graph..." << endl << endl;
    hybrid* hybrid_graph;
    hybrid_graph =new hybrid(points->GetDimofPoints(), points, graph_path, 100, 100, 16, 64);
    // cout << "Train RVQ..." << endl;
	// hybrid_graph->hybrid_train(points->GetFirstPositionofPoint(0), points->GetNumPoints());
    // cout << "Build RVQ..." << endl;
    // hybrid_graph->hybrid_build(points->GetFirstPositionofPoint(0), points->GetNumPoints());
    // cout << "Save RVQ..." << endl;
    // hybrid_graph->hybrid_save("/home/ErHa/GANNS_Res/rvq_model_100_100_500000_1M.bin");
    cout << "Load RVQ..." << endl;
    hybrid_graph->hybrid_load("/home/ErHa/GANNS_Res/rvq_model_100_100_1M.bin");

    //for(int i = 0; i < 2; i++){
    int* results = NULL;
    cout << "Search...Query number: " <<query_points->GetNumPoints()<<" Num of candidates: "<<num_of_candidates<<endl << endl;
    hybrid_graph->hybrid_search(query_points->GetFirstPositionofPoint(0), num_of_topk, results, query_points->GetNumPoints(), num_of_candidates);
    
    float recall = 0;
    
    ComputeRecall(results, groundtruth, query_points->GetNumPoints(), num_of_topk, k_of_groundtruth, pow(2.0, ceil(log(num_of_topk) / log(2))), recall);
    cout << "Recall: " << recall << endl;
    //}

    return 0;
}
