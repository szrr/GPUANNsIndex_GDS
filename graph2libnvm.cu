#include<iostream>
#include"bam_impl.h"
#include "read_write.cuh"

Parameters parameters;
std::vector<Controller*> ctrls;
page_cache_t* h_pc;

range_t<_Float32>* dg_range;

std::vector<range_t<_Float32>*> dg;

array_t<_Float32>* diskgraph;

const char* const ctrls_paths[] = {"/dev/libnvm0"};

void init_page_cache(uint64_t n_items, uint64_t len_item)   // （向量的个数，每个item的元素个数）
{
    int cnt = 0;
    ctrls.resize(parameters.n_ctrls);
    cuda_err_chk(cudaSetDevice(parameters.cudaDevice));
    printf("Queue Depth %lu\n", parameters.queueDepth);
    for (uint32_t i = 0; i < parameters.n_ctrls; i++) {
        ctrls[i] = new Controller(ctrls_paths[i], parameters.nvmNamespace, parameters.cudaDevice, parameters.queueDepth, parameters.numQueues);
    }
    printf("Controllers Created.\n");

    uint64_t pc_page_size = parameters.pageSize;                                  // 单个缓存页大小
    uint64_t pc_pages = ceil((float)parameters.maxPageCacheSize/pc_page_size);    // 缓存页数量（槽数）

    printf("Initialization of PageCache on Device DONE. \n");
    fflush(stdout);

    uint64_t dgSize = n_items*(sizeof(_Float32)*len_item);            // 全图的元素数
    uint64_t n_pages = (dgSize+pc_page_size-1) / pc_page_size;    // 全图可被分成的页数
    uint64_t n_dgElements = n_items*len_item;
    
    h_pc = new page_cache_t(pc_page_size, pc_pages, parameters.cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
    
    dg_range = new range_t<_Float32>(0, n_dgElements, 0, n_pages, 0, pc_page_size, h_pc, parameters.cudaDevice);
    dg.resize(1);
    dg[0] = dg_range;
    diskgraph = new array_t<_Float32>(n_dgElements, 0, dg, parameters.cudaDevice, cnt++);
    

    printf("Page Cache Initialized\n");
    fflush(stdout);

}

void reclaim_page_cache() 
{
    diskgraph->print_reset_stats();
    delete diskgraph;
    delete dg_range;

    ctrls[0]->print_reset_stats();
    delete ctrls[0];
#if USE_HOST_CACHE
    revokeHostRuntime();
#endif
}

void process() {
    uint64_t n_items = parameters.n_items;
    uint64_t len_item = parameters.len_item;
    uint64_t dim = parameters.dim;
    std::string binpath = parameters.binpath;
    int cudaDeviceId = parameters.cudaDeviceId;
    uint32_t threadsPerBlock = 256;
    uint32_t blocksPerGrid = 8;

    printf("n_items = %llu\n",n_items);

    
    init_page_cache(n_items,len_item);

    float *data;
    cudaMallocManaged(&data, n_items*dim*sizeof(_Float32));
    std::ifstream file(binpath, std::ios::binary);
    uint32_t D; // D暂时没啥用，需要用的时候再放到for里面去
    if (file.is_open()) {
        size_t float_offset = 0;  // 用于追踪浮点数在统一内存中的偏移量
        for (uint32_t i = 0; i < n_items; ++i) {
            
            file.read(reinterpret_cast<char*>(&D), sizeof(uint32_t));  // 读取整数 D
            file.read(reinterpret_cast<char*>(data + float_offset), dim * sizeof(float));  // 读取 D 个浮点数
            float_offset += dim;  // 更新偏移量
        }
        file.close();
    } else {
        std::cerr << "Failed to open the file" << std::endl;
    }

    cuda_err_chk(cudaSetDevice(cudaDeviceId));

    float time_elapsed;
    cudaEvent_t start, end;
    cuda_err_chk(cudaEventCreate(&start));
    cuda_err_chk(cudaEventCreate(&end));
    cuda_err_chk(cudaEventRecord(start, 0));



    // writePreGraph2SSD<<<blocksPerGrid, threadsPerBlock>>>(diskgraph->d_array_ptr, data, n_items);
    readTest<<<blocksPerGrid, threadsPerBlock>>>(diskgraph->d_array_ptr, n_items);
    
    

    h_pc->flush_cache();

    cuda_err_chk(cudaDeviceSynchronize());
    cuda_err_chk(cudaEventRecord(end, 0));
    cuda_err_chk(cudaEventSynchronize(end));
    cuda_err_chk(cudaEventElapsedTime(&time_elapsed, start, end));
    printf("Elapsed Time %f ms\n", time_elapsed);
	printf("Computation Done\n");


    reclaim_page_cache();
    printf("freeing...");
    cudaFree(data);

}


int main(){
    process();
    return 0;
}