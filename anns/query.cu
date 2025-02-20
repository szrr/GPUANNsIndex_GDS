#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "hybrid/hybrid.h"
#include "./graph/graph_kernel_operation/bamWriteSSD.cuh"


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
    // cudaMallocHost(&groundtruth, num_of_points * k_of_groundtruth * sizeof(int));
    // groundtruth = new int[num_of_points * k_of_groundtruth];
    if (cudaMallocHost(&groundtruth, num_of_points * k_of_groundtruth * sizeof(int)) != cudaSuccess) {
        cout << "Memory allocation failed." << endl;
        exit(-1);
    }

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
            int* position_of_candidate = NULL;
            // position_of_candidate = find(groundtruth + i * k_of_groundtruth, groundtruth + i * k_of_groundtruth + num_of_topk, crt_candidate_id);
            position_of_candidate = find(groundtruth + i * k_of_groundtruth, groundtruth + i * k_of_groundtruth + num_of_topk, crt_candidate_id);
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

// template <typename T>
// void read_data_point(const std::string& filepath, size_t target_id, size_t dim, size_t degree, size_t numElementsPerBlock) {
//     // 打开文件
//     std::ifstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open the file!" << std::endl;
//         return;
//     }
//     size_t page_size = 4096;
//     size_t data_point_size = (dim + degree) * sizeof(T);

//     size_t target_offset = (target_id / numElementsPerBlock) * page_size + (target_id % numElementsPerBlock) * (dim + degree) * sizeof(T);

//     // 将文件指针移动到目标数据点的位置
//     file.seekg(target_offset, std::ios::beg);
//     if (!file) {
//         std::cerr << "Error seeking to the desired position!" << std::endl;
//         file.close();
//         return;
//     }

//     std::vector<char> elements((dim + degree) * sizeof(T));
//     file.read(reinterpret_cast<char*>(elements.data()), data_point_size);
//     printf("Offste: %lu\n", target_offset / sizeof(T));
//     // 输出读取的n个元素
//     if (file) {
//         std::cout << "Data for ID " << target_id << ": ";
//         for (int i = 0; i < dim; i++) { 
//             float* data = reinterpret_cast<float*>(elements.data());
//             printf("%f ", data[i]);
//         }
//         std::cout << std::endl;
//         std::cout << "Neighbors for ID " << target_id << ": ";
//         for(int i = 0; i < degree; i++){
//             int* neighbors = reinterpret_cast<int*>(elements.data() + dim * sizeof(float));
//             printf("%d ", neighbors[i]);
//         }
//         std::cout << std::endl;
//     } else {
//         std::cerr << "Error reading data!" << std::endl;
//     }

//     // 文件关闭
//     file.close();
// }
template <typename T>
void read_data_point(const std::string& filepath, size_t target_id, size_t n, size_t offset) {
    // 打开文件
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
        return;
    }
    // 计算目标数据点的偏移量
    // 假设文件中的每个数据点由n个元素组成，元素类型为T
    // 每个数据点的大小为 n * sizeof(T)
    size_t data_point_size = n * sizeof(T);

    // 计算目标数据点的位置
    size_t target_offset = offset + target_id * data_point_size;

    // 将文件指针移动到目标数据点的位置
    file.seekg(target_offset, std::ios::beg);
    if (!file) {
        std::cerr << "Error seeking to the desired position!" << std::endl;
        file.close();
        return;
    }

    // 读取n个元素
    std::vector<T> elements(n);
    file.read(reinterpret_cast<char*>(elements.data()), data_point_size);
    printf("Offste: %lu\n", target_offset / sizeof(T));
    // 输出读取的n个元素
    if (file) {
        std::cout << "Data for ID " << target_id << ": ";
        for (const auto& elem : elements) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error reading data!" << std::endl;
    }

    // 文件关闭
    file.close();
}

__global__ void read_test(array_d_t<char> *ssd_data, size_t n, size_t offset){
    size_t t_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t t_num = gridDim.x * blockDim.x;
    for(size_t i = 0; i < (n + t_num - 1) / t_num; i++){
        size_t idx = t_id + i * t_num;
        if(idx < n){
            char target = (*ssd_data)[idx * offset];
        }
    }
}

int main(int argc,char** argv){
    cudaSetDevice(3);
    //required variables from external input
    string base_path = argv[1];
    string query_path = argv[2];
    string graph_path = argv[3];
    string groundtruth_path = argv[4];
    int num_of_candidates = atoi(argv[5]);
    int search_width = atoi(argv[6]);
    int num_of_topk = atoi(argv[7]);
    int num_of_points = atoi(argv[8]);
    int dim_of_points = atoi(argv[9]);
    int degree_of_graph = atoi(argv[10]);
    size_t max_page_size = stoul(argv[11]);
    size_t num_of_queues = stoul(argv[12]);
    size_t depth_of_queue = stoul(argv[13]);

    // cout << "Load groundtruth..." << endl;
    int* groundtruth = NULL;
    int k_of_groundtruth;
    LoadGroundtruth(groundtruth, groundtruth_path, k_of_groundtruth);

    cout << "Load data points and query points..." << endl;
    // Data* points = new Data(base_path,num_of_points);
	
    Data* query_points = new Data(query_path);
    cout << "Search...Query number: " <<query_points->GetNumPoints()<<" Num of candidates: "<<num_of_candidates<<endl;

    Settings settings;
    settings.maxPageCacheSize = max_page_size;
    settings.numQueues = num_of_queues;
    settings.queueDepth = depth_of_queue;
    std::vector<Controller *> ctrls(settings.n_ctrls);
    const char *const ctrls_paths[] = {"/dev/libnvm0"};
    ctrls[0] = new Controller(ctrls_paths[0], settings.nvmNamespace,
                              settings.cudaDevice, settings.queueDepth,
                              settings.numQueues);

    // initialize GPU page cacahe
    uint64_t pc_pages =  settings.maxPageCacheSize / settings.pageSize;
    printf("pageSize %d, pageNum %d, cudaDevice %d\n", settings.pageSize, pc_pages, settings.cudaDevice);
    page_cache_t h_pc(settings.pageSize, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t)16, ctrls);
    page_cache_d_t *d_pc = (page_cache_d_t *)(h_pc.d_pc_ptr);
    size_t elementSize = dim_of_points * sizeof(float) + degree_of_graph * sizeof(int) + sizeof(int);
    size_t numElementsPerBlock = settings.pageSize / elementSize;
    size_t remainingSpace = settings.pageSize - numElementsPerBlock * elementSize;
    size_t data_pages = (num_of_points + numElementsPerBlock - 1) / numElementsPerBlock;
    size_t totalSize = settings.pageSize * data_pages;
    size_t num_of_data_elements = totalSize / sizeof(char);
    
    // size_t elementSize = dim_of_points * sizeof(float) + degree_of_graph * sizeof(int) + sizeof(int);
    // size_t numElementsPerBlock = 4096 / elementSize;
    // size_t data_pages = ((num_of_points + numElementsPerBlock - 1) / numElementsPerBlock);
    // size_t totalSize = 4096 * data_pages;
    // data_pages /= (settings.pageSize / 4096);
    // size_t num_of_data_elements = totalSize / sizeof(char);
    
    // writeArray2Disk<char>(base_path, num_of_data_elements, 0, h_pc, d_pc, settings);

    range_t<char> *data_range = new range_t<char>(
        (uint64_t)0, num_of_data_elements, 0, data_pages, 
        (uint64_t)0, settings.pageSize, &h_pc, settings.cudaDevice);


    std::vector<range_t<char> *> vec_datarange(1);

    vec_datarange[0] = data_range;

    array_t<char> *d_data_array = new array_t<char>(num_of_data_elements, 0, vec_datarange, settings.cudaDevice);
    // size_t n = 1000000;
    // Timer a;
    // a.Start();
    // read_test<<<10000, 32>>>(d_data_array->d_array_ptr, n, settings.pageSize);
    // cudaDeviceSynchronize();
    // a.Stop();
    // cout << "Read " << n << " points, page size: " << settings.pageSize << ", use " << a.DurationInMilliseconds() << " ms" << endl;
    // size_t test_id = 0;//655046862; //99999999
    // size_t test_dim = size_t(dim_of_points);
    // size_t test_nei = size_t(degree_of_graph);
    // std::ifstream in("/mnt/Samsung980PRO2TB/szr/starling/bigann_100m_M256_R64_L100_B256/starlingIndex.bin", std::ios::binary);
    // in.seekg(test_id * sizeof(int), std::ios::beg);
    // int tmp_test_id;
    // in.read((char*)(&tmp_test_id), sizeof(int));
    // in.close();
    // printf("%lu : %lu\n", test_id, tmp_test_id);
    // read_data_kernel<<<1,test_dim + test_nei>>>(d_data_array->d_array_ptr, size_t(tmp_test_id), test_dim, test_nei, numElementsPerBlock);
    // CUDA_SYNC_CHECK();
    // read_data_point<float>("/data/szr/diskann/sift1b/SIFT1B_DATASET.bin", test_id, test_dim, 8);
    // cout << "Load proximity graph..." << endl;
    Data* points = new Data(num_of_points, dim_of_points);
    hybrid* hybrid_graph;
    hybrid_graph =new hybrid(128, points, graph_path, 1000, 100, degree_of_graph, 64);
    // cout << "Train RVQ..." << endl;
	// hybrid_graph->hybrid_train(base_path, num_of_points);
    // hybrid_graph->hybrid_loadCodebook("/home/ErHa/GANNS_Res/rvq/Codebook_1000_100_500000_100M.bin");
    // cout << "Build RVQ..." << endl;
    // hybrid_graph->hybrid_build(base_path, num_of_points);
    // cout << "Save RVQ..." << endl;
    // hybrid_graph->hybrid_save("/home/ErHa/GANNS_Res/rvq/rvq_model_1000_100_500000_100M.bin");

    cout << "Load RVQ..." << endl;
    hybrid_graph->hybrid_load("/home/ErHa/GANNS_Res/rvq/rvq_model_1000_100_500000_100M.bin");

    int* results = NULL;
    // cout << "Search...Query number: " <<query_points->GetNumPoints()<<" Num of candidates: "<<num_of_candidates<<endl;
    hybrid_graph->hybrid_search(d_data_array, query_points->GetFirstPositionofPoint(0), num_of_topk, results, query_points->GetNumPoints(), num_of_candidates, search_width);
    
    float recall = 0;
    ComputeRecall(results, groundtruth, query_points->GetNumPoints(), num_of_topk, k_of_groundtruth, num_of_topk, recall);
    cout << "Recall: " << recall << endl;
    d_data_array->print_reset_stats();
    // ctrls[0]->print_reset_stats();


    return 0;
}
