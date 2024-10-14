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
            // position_of_candidate = find(groundtruth + i * k_of_groundtruth, groundtruth + i * k_of_groundtruth + num_of_topk, crt_candidate_id);
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

void readData(string path, int dim, int num, float* data, int start_loc){
    ifstream in_descriptor(path, std::ios::binary);
    
    if (!in_descriptor.is_open()) {
        exit(1);
    }
    //in_descriptor.seekg(0, std::ios::beg);
    for (int i = 0; i < num; i++) {
        unsigned char tmp_data[dim];
        in_descriptor.seekg(4, std::ios::cur);
        in_descriptor.read((char*)(tmp_data), dim);
        for(int l = 0;l < dim; l++){
            data[start_loc * dim + i * dim + l] = float(tmp_data[l]);
        }
    }

    in_descriptor.close();
}

void readSubgraphData(float* data, int dim, int num_of_subgraph, int* pre_fix_of_subgraph_size){
    for(int i = 0; i < num_of_subgraph; i++){
        ostringstream data_filename;
        data_filename << "subData" << std::setw(4) << std::setfill('0') << i << ".bin";
        string data_path = "/home/ErHa/GANNS_Res/subdata/" + data_filename.str();
        readData(data_path, dim, pre_fix_of_subgraph_size[i+1] - pre_fix_of_subgraph_size[i] ,data, pre_fix_of_subgraph_size[i]);
    }
}

void writeBinaryFile(float* data, int n, int dim, const std::string& filename) {

    // 打开二进制文件，ios::binary 表示以二进制方式打开
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 写入数据：对于每个数据点，先写入维度（dim），再写入 dim 个 float 数据
    outFile.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(data), n * dim * sizeof(float));
    // for (int i = 0; i < n; ++i) {
    //     // 写入维度（dim）
    //     outFile.write(reinterpret_cast<const char*>(&dim), sizeof(int));

    //     // 写入 dim 个 float 数据
    //     outFile.write(reinterpret_cast<const char*>(&data[i * dim]), dim * sizeof(float));
    // }

    // 关闭文件
    outFile.close();
}

int main(int argc,char** argv){
    // cudaSetDevice(1);
    //required variables from external input
    string base_path = argv[1];
    string query_path = argv[2];
    string graph_path = argv[3];
    string groundtruth_path = argv[4];
    int num_of_candidates = atoi(argv[5]);
    int search_width = atoi(argv[6]);
    int num_of_topk = atoi(argv[7]);
    int num_of_points = atoi(argv[8]);
    int degree_of_graph = atoi(argv[9]);

    cout << "Load groundtruth..." << endl << endl;
    int* groundtruth = NULL;
    int k_of_groundtruth;
    LoadGroundtruth(groundtruth, groundtruth_path, k_of_groundtruth);

    cout << "Load data points and query points..." << endl << endl;
    Data* points = new Data(base_path,num_of_points);
    // writeBinaryFile(points->data_, num_of_points, 128, "/home/ErHa/GANNS_Res/SIFT1M");
    // for(int i = 0; i < 10; i++){
    //     for(int l = 0; l < 128; l++){
    //         printf("%.0f ",points->data_[i * 128 + l]);
    //     }
    //     printf("\n\n");
    // }
	// string pre_fix_of_subgraph_size_path = "/home/ErHa/GANNS_Res/subdata/pre_fix_of_subgraph_size.bin";
    // std::ifstream in(pre_fix_of_subgraph_size_path, std::ios::binary);
    // int num_of_subgraph;
    // in.read(reinterpret_cast<char*>(&num_of_subgraph), sizeof(int));
    // int* pre_fix_of_subgraph_size = new int[num_of_subgraph + 1];
    // in.read(reinterpret_cast<char*>(pre_fix_of_subgraph_size), (num_of_subgraph + 1) * sizeof(int));
    // in.close();
    // readSubgraphData(points->data_, 128, num_of_subgraph, pre_fix_of_subgraph_size);
    // std::ifstream inn("/home/ErHa/GANNS_Res/subdata/new_index_of_data.bin", std::ios::binary);
    // int num_of_index;
    // inn.read(reinterpret_cast<char*>(&num_of_index), sizeof(int));
    // int* new_index_of_data = new int[num_of_index];
    // inn.read(reinterpret_cast<char*>(new_index_of_data), num_of_index * sizeof(int));
    // inn.close();
    // for(int i = 0; i < 10; i++){
    //     for(int k = 0; k < num_of_index; k++){
    //         if(new_index_of_data[k] == i){
    //             for(int l = 0; l < 128; l++){
    //                 printf("%.0f ",points->data_[k * 128 + l]);
    //             }
    //             printf("\n\n");
    //         }
    //     }
    // }
    Data* query_points = new Data(query_path);

    cout << "Load proximity graph..." << endl << endl;
    hybrid* hybrid_graph;
    hybrid_graph =new hybrid(points->GetDimofPoints(), points, graph_path, 100, 100, degree_of_graph, 64); // points points->GetDimofPoints()
    // cout << "Train RVQ..." << endl;
	// hybrid_graph->hybrid_train();
    // hybrid_graph->hybrid_loadCodebook("/home/ErHa/GANNS_Res/rvq/Codebook_250_250_100000_10M.bin");
    // cout << "Build RVQ..." << endl;
    // hybrid_graph->hybrid_build("/mnt/data1/szr/dataset/sift1b/bigann_base.bvecs", num_of_points);
    // cout << "Save RVQ..." << endl;
    // hybrid_graph->hybrid_save("/home/ErHa/GANNS_Res/rvq/rvq_model_250_250_100000_10M.bin");
    // cout << "Load RVQ..." << endl;
    // hybrid_graph->hybrid_load("/home/ErHa/GANNS_Res/rvq/rvq_model_500_500_1000000_10M.bin");

    int* results = NULL;
    cout << "Search...Query number: " <<query_points->GetNumPoints()<<" Num of candidates: "<<num_of_candidates<<endl << endl;
    hybrid_graph->hybrid_search(query_points->GetFirstPositionofPoint(0), num_of_topk, results, query_points->GetNumPoints(), num_of_candidates, search_width);
    
    float recall = 0;
    // pow(2.0, ceil(log(num_of_topk) / log(2)))
    ComputeRecall(results, groundtruth, query_points->GetNumPoints(), num_of_topk, k_of_groundtruth, num_of_topk, recall);
    cout << "Recall: " << recall << endl;

    return 0;
}
