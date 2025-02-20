#pragma once

#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <byteswap.h>

#include<math.h>
#include<cuda_runtime.h>
#include "structure_on_device.cuh"
// #define size32 153600
// #define size32 262144
#define size32 1024
#define tmp_size32 25600
// #define shift 9
const int shift = 9;
// #define num_hash 3
const int num_hash = 3;

__device__
int hash_(int h,uint32_t x,uint32_t* random_number){ 
    x ^= x >> 16;
    x *= random_number[h << 1];
    x ^= x >> 13;
    x *= random_number[(h << 1) + 1];
    x ^= x >> 16;
    //return x & 31;
    return x % ((size32 << 5));
}

__device__
void set_bit(uint32_t x,uint32_t* data){
    unsigned int mask = 1U << (x / size32);
    atomicOr(&data[x % size32], mask);
}

__device__
bool test_bit(int offset,uint32_t x,uint32_t* data){
    //return ((data[offset] >> x) & 1);
    return ((data[x % size32] >> (x / size32)) & 1);
}


__device__
void add(uint32_t x,uint32_t* random_number,uint32_t* data){
    for(int i = 0;i < num_hash;++i)
        set_bit(hash_(i,x,random_number),data);
}

__device__
bool test(uint32_t x,uint32_t* random_number,uint32_t* data){
    int offset = 0; //get_offset(x,random_number);
    bool ok = true;
    for(int i = 0;i < num_hash;++i)
        ok &= test_bit(offset,hash_(i,x,random_number),data);
    return ok;
}

__device__
int tmp_hash_(int h,uint32_t x,uint32_t* random_number){ 
    x ^= x >> 16;
    x *= random_number[h << 1];
    x ^= x >> 13;
    x *= random_number[(h << 1) + 1];
    x ^= x >> 16;
    //return x & 31;
    return x % ((tmp_size32 << 5));
}

__device__
void tmp_set_bit(uint32_t x,uint32_t* data){
    unsigned int mask = 1U << (x / tmp_size32);
    atomicOr(&data[x % tmp_size32], mask);
}

__device__
bool tmp_test_bit(int offset,uint32_t x,uint32_t* data){
    //return ((data[offset] >> x) & 1);
    return ((data[x % tmp_size32] >> (x / tmp_size32)) & 1);
}


__device__
void tmp_add(uint32_t x,uint32_t* random_number,uint32_t* data){
    for(int i = 0;i < num_hash;++i)
        tmp_set_bit(tmp_hash_(i,x,random_number),data);
}

__device__
bool tmp_test(uint32_t x,uint32_t* random_number,uint32_t* data){
    int offset = 0; //get_offset(x,random_number);
    bool ok = true;
    for(int i = 0;i < num_hash;++i)
        ok &= tmp_test_bit(offset,tmp_hash_(i,x,random_number),data);
    return ok;
}

template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void cagra(float* d_data, float* d_query, KernelPair<float, int>* d_result, int* d_graph, int total_num_of_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int search_width,
                    int* d_enter_cluster, int** d_rvq_indices, int* d_rvq_indices_size, size_t hash_size, uint32_t* d_hash_table, int* starling_index){
    #define DIM 128
    int t_id = threadIdx.x;
    size_t b_id = size_t(blockIdx.x);
    int size_of_warp = 32;
    int size_of_block = blockDim.x;
    size_t lane_id = size_t(threadIdx.x) % size_t(size_of_warp);
    int warp_id = threadIdx.x / size_of_warp;
    
    KernelPair<float, int>* crt_result = d_result + b_id * num_of_results;
    int degree_of_point = (1 << offset_shift);
    int num_of_visited_points_one_batch = (search_width << offset_shift);
    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch;
    }

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;

    // __shared__ uint32_t data[tmp_size32];
    uint32_t* data = d_hash_table + b_id * hash_size;
    uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
    };

#if DIM > 0
	float q1 = 0;
	if (lane_id < DIM) {
		q1 = d_query[b_id * DIM + lane_id];
	}
#endif
#if DIM > 32
    float q2 = 0;
    if (lane_id + 32 < DIM) {
        q2 = d_query[b_id * DIM + lane_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (lane_id + 64 < DIM) {
    	q3 = d_query[b_id * DIM + lane_id + 64];
   	}
#endif
#if DIM > 96
    float q4 = 0;
    if (lane_id + 96 < DIM) {
    	q4 = d_query[b_id * DIM + lane_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (lane_id + 128 < DIM) {
        q5 = d_query[b_id * DIM + lane_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (lane_id + 160 < DIM) {
        q6 = d_query[b_id * DIM + lane_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (lane_id + 192 < DIM) {
        q7 = d_query[b_id * DIM + lane_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (lane_id + 224 < DIM) {
        q8 = d_query[b_id * DIM + lane_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (lane_id + 256 < DIM) {
        q9 = d_query[b_id * DIM + lane_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (lane_id + 288 < DIM) {
        q10 = d_query[b_id * DIM + lane_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (lane_id + 320 < DIM) {
        q11 = d_query[b_id * DIM + lane_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (lane_id + 352 < DIM) {
        q12 = d_query[b_id * DIM + lane_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (lane_id + 384 < DIM) {
        q13 = d_query[b_id * DIM + lane_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (lane_id + 416 < DIM) {
        q14 = d_query[b_id * DIM + lane_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (lane_id + 448 < DIM) {
        q15 = d_query[b_id * DIM + lane_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (lane_id + 480 < DIM) {
        q16 = d_query[b_id * DIM + lane_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (lane_id + 512 < DIM) {
        q17 = d_query[b_id * DIM + lane_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (lane_id + 544 < DIM) {
        q18 = d_query[b_id * DIM + lane_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (lane_id + 576 < DIM) {
        q19 = d_query[b_id * DIM + lane_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (lane_id + 608 < DIM) {
        q20 = d_query[b_id * DIM + lane_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (lane_id + 640 < DIM) {
        q21 = d_query[b_id * DIM + lane_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (lane_id + 672 < DIM) {
        q22 = d_query[b_id * DIM + lane_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (lane_id + 704 < DIM) {
        q23 = d_query[b_id * DIM + lane_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (lane_id + 736 < DIM) {
        q24 = d_query[b_id * DIM + lane_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (lane_id + 768 < DIM) {
        q25 = d_query[b_id * DIM + lane_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (lane_id + 800 < DIM) {
        q26 = d_query[b_id * DIM + lane_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (lane_id + 832 < DIM) {
        q27 = d_query[b_id * DIM + lane_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (lane_id + 864 < DIM) {
        q28 = d_query[b_id * DIM + lane_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (lane_id + 896 < DIM) {
        q29 = d_query[b_id * DIM + lane_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (lane_id + 224 < DIM) {
        q30 = d_query[b_id * DIM + lane_id + 928];
    }
#endif
    
//Search    
    //First iteration
    // int cluster_id = d_enter_cluster[b_id];
    // int enter_points_num = min(d_rvq_indices_size[cluster_id], 2 * num_of_visited_points_one_batch);
    int enter_points_num = 2 * num_of_visited_points_one_batch;
    // int* enter_points_pos = d_rvq_indices[cluster_id];
    
    int iteration;
    //int enter_points_num = NumThreadQ * size_of_warp;
    int step_id;
    int substep_id;
    
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;

        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {

            neighbors_array[unrollt_id].first = MAX;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }
    __syncthreads();
    iteration = (enter_points_num + num_of_visited_points_one_batch -1) / num_of_visited_points_one_batch;
    for(int iter = 0; iter < iteration; iter++){
        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
            int unrollt_id = t_id + size_of_block * i;

            if (unrollt_id < num_of_visited_points_one_batch && iter * num_of_visited_points_one_batch + unrollt_id < enter_points_num) { //enter_points_num
                
                // neighbors_array[num_of_candidates + unrollt_id].second = enter_points_pos[iter * num_of_visited_points_one_batch + unrollt_id];
                neighbors_array[num_of_candidates + unrollt_id].second = iter * num_of_visited_points_one_batch + unrollt_id;
                
                add(starling_index[neighbors_array[num_of_candidates + unrollt_id].second], random_number, data);
            }
            else if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
            }
        }
        __syncthreads();
        for (int i = warp_id; i < num_of_visited_points_one_batch; i+=2) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);
            
            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = float(d_data[target_point_id * DIM + lane_id]);
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = float(d_data[target_point_id * DIM + lane_id + 32]);
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = float(d_data[target_point_id * DIM + lane_id + 64]);
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = float(d_data[target_point_id * DIM + lane_id + 96]);
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float(d_data[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float(d_data[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float(d_data[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float(d_data[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float(d_data[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float(d_data[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float(d_data[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float(d_data[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float(d_data[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float(d_data[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float(d_data[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float(d_data[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float(d_data[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float(d_data[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float(d_data[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float(d_data[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float(d_data[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float(d_data[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float(d_data[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float(d_data[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float(d_data[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float(d_data[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float(d_data[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float(d_data[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float(d_data[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float(d_data[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif
            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;
            }

        }
        
        __syncthreads();
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block-1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if ((neighbors_array[unrollt_id].first) > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                    
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        
    }
    __syncthreads();
    
    // for(int i = 0; i < (num_of_candidates * num_hash + size_of_block - 1) / size_of_block; i++){
    //     int unrollt_id = t_id + i * size_of_block;
    //     if(unrollt_id < num_of_candidates * num_hash){
    //         int index = neighbors_array[unrollt_id / num_hash].second;
    //         if(index < total_num_of_points){
    //             set_bit(hash_(unrollt_id % num_hash, index, random_number),data);
    //         }
    //     }
    // }
    // __syncthreads();
    int flag_all_blocks = 1;
    int crt_flag = 0;
    int tmp_flag = (1 << min(min(num_of_explored_points, size_of_warp), enter_points_num)) - 1;;

    int tmp_search_width;
    int first_position_of_flag;
    int check_zero = 0;
    int hash_iteration = 0;
    iteration = 0;
    int iter = 0;
    int max_iter = num_of_explored_points / search_width;
    while (flag_all_blocks && iter < max_iter){
        iter++;
        
        for(int i = 0 ; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++){
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();
        tmp_search_width = 0;
        while(tmp_search_width < search_width && tmp_flag != 0){

            int first_position_of_tmp_flag = __ffs(tmp_flag) - 1;
            int neighbor_loc = iteration * size_of_warp + first_position_of_tmp_flag;
            
            first_position_of_flag = abs(neighbors_array[neighbor_loc].second);
            
            
            if(t_id == 0){
                neighbors_array[neighbor_loc].second = -neighbors_array[neighbor_loc].second;
            }
            tmp_flag &= ~(1 << first_position_of_tmp_flag);

            //读取邻居
            auto offset = size_t(first_position_of_flag) << offset_shift;
            int* neighbors_of_first_point = d_graph + offset;
            for (int i = 0; i < (degree_of_point + size_of_block - 1) / size_of_block; i++) {
                int unrollt_id = t_id + size_of_block * i;
                if (unrollt_id < degree_of_point) {
                    int target_point = neighbors_of_first_point[unrollt_id]; 
                    if(target_point == total_num_of_points || test(starling_index[target_point], random_number, data)){
                        neighbors_array[num_of_candidates + unrollt_id + tmp_search_width * degree_of_point].second = total_num_of_points;
                    }else{
                        neighbors_array[num_of_candidates + unrollt_id + tmp_search_width * degree_of_point].second = target_point; 
                        add(starling_index[target_point], random_number, data);
                    }
                    
                }
            }
            __syncthreads();

            
            tmp_search_width++;
            if(tmp_search_width == search_width) break;

            while(tmp_flag == 0 && iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp){
                iteration++;
                int unrollt_id = lane_id + size_of_warp * iteration;
                crt_flag = 0;
                if(unrollt_id < num_of_explored_points){
                    if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                        crt_flag = 1;
                    }else if(neighbors_array[unrollt_id].second == 0){
                        if(check_zero == 0){
                            check_zero = 1;
                            crt_flag = 1;
                        }
                    }
                }
                tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
            }
            
        }
        __syncthreads();
        if(tmp_search_width == 0) break;
        //计算query与search_width个点的邻居的距离
            

        for (int i = warp_id; i < tmp_search_width * degree_of_point; i+=2) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);

            if (target_point_id >= total_num_of_points) {
                neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            
            
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = float(d_data[target_point_id * DIM + lane_id]);
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = float(d_data[target_point_id * DIM + lane_id + 32]);
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = float(d_data[target_point_id * DIM + lane_id + 64]);
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = float(d_data[target_point_id * DIM + lane_id + 96]);
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float(d_data[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float(d_data[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float(d_data[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float(d_data[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float(d_data[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float(d_data[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float(d_data[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float(d_data[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float(d_data[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float(d_data[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float(d_data[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float(d_data[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float(d_data[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float(d_data[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float(d_data[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float(d_data[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float(d_data[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float(d_data[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float(d_data[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float(d_data[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float(d_data[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float(d_data[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float(d_data[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float(d_data[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float(d_data[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float(d_data[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif

            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;    
            }

        }
        __syncthreads();
        // 对邻居列表排序
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        //将候选列表与邻居列表merge之后排序
        
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if(neighbors_array[unrollt_id + num_of_visited_points_one_batch].first == MAX) continue;
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        // //判断循环条件

        for (iteration = 0; iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp; iteration++) {
            int unrollt_id = lane_id + size_of_warp * iteration;
            crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                //crt_flag = flags[unrollt_id];
                if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                    crt_flag = 1;
                }else if(neighbors_array[unrollt_id].second == 0){
                    if(check_zero == 0){
                        check_zero = 1;
                        crt_flag = 1;
                    }
                }
            }
            tmp_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(tmp_flag != 0){
                break;
            }else if(iteration == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }
        
        // if(hash_iteration == 4){
        //     for(int i = 0; i < (tmp_size32 + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < tmp_size32){
        //             data[unrollt_id] = 0;
        //         }
        //     }
        //     __syncthreads();
            
        //     for(int i = 0; i < (num_of_candidates * num_hash + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < num_of_candidates * num_hash){
        //             int index = abs(neighbors_array[unrollt_id / num_hash].second);
        //             if(index < total_num_of_points){
        //                tmp_set_bit(tmp_hash_(unrollt_id % num_hash,index,random_number),data); 
        //             }
        //         }
        //     }
        //     //__syncthreads();
        //     hash_iteration = 0;
        // }
       
        hash_iteration++;
    }
    
    for (int i = 0; i < (num_of_results + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;
    
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id].second = abs(neighbors_array[unrollt_id].second);
            crt_result[unrollt_id].first = neighbors_array[unrollt_id].first;
        }
    }
}

template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void cagra_block(float* d_data, float* d_query, KernelPair<float, int>* d_result, int* d_graph, int total_num_of_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int search_width,
                    int* d_enter_cluster, int** d_rvq_indices, int* d_rvq_indices_size, size_t hash_size, uint32_t* d_hash_table, int* starling_index,
                    int num_elements_per_block, int length_of_block){
    #define DIM 128
    int t_id = threadIdx.x;
    size_t b_id = size_t(blockIdx.x);
    int size_of_warp = 32;
    int size_of_block = blockDim.x;
    size_t lane_id = size_t(threadIdx.x) % size_t(size_of_warp);
    int warp_id = threadIdx.x / size_of_warp;
    
    KernelPair<float, int>* crt_result = d_result + b_id * num_of_results;
    int degree_of_point = (1 << offset_shift);
    int num_of_visited_points_one_batch = (search_width << offset_shift) * length_of_block;
    int length_of_visited_points_one_batch = (search_width << offset_shift);
    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch;
    }

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;

    // __shared__ uint32_t data[tmp_size32];
    uint32_t* data = d_hash_table + b_id * hash_size;
    uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
    };

#if DIM > 0
	float q1 = 0;
	if (lane_id < DIM) {
		q1 = d_query[b_id * DIM + lane_id];
	}
#endif
#if DIM > 32
    float q2 = 0;
    if (lane_id + 32 < DIM) {
        q2 = d_query[b_id * DIM + lane_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (lane_id + 64 < DIM) {
    	q3 = d_query[b_id * DIM + lane_id + 64];
   	}
#endif
#if DIM > 96
    float q4 = 0;
    if (lane_id + 96 < DIM) {
    	q4 = d_query[b_id * DIM + lane_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (lane_id + 128 < DIM) {
        q5 = d_query[b_id * DIM + lane_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (lane_id + 160 < DIM) {
        q6 = d_query[b_id * DIM + lane_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (lane_id + 192 < DIM) {
        q7 = d_query[b_id * DIM + lane_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (lane_id + 224 < DIM) {
        q8 = d_query[b_id * DIM + lane_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (lane_id + 256 < DIM) {
        q9 = d_query[b_id * DIM + lane_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (lane_id + 288 < DIM) {
        q10 = d_query[b_id * DIM + lane_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (lane_id + 320 < DIM) {
        q11 = d_query[b_id * DIM + lane_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (lane_id + 352 < DIM) {
        q12 = d_query[b_id * DIM + lane_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (lane_id + 384 < DIM) {
        q13 = d_query[b_id * DIM + lane_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (lane_id + 416 < DIM) {
        q14 = d_query[b_id * DIM + lane_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (lane_id + 448 < DIM) {
        q15 = d_query[b_id * DIM + lane_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (lane_id + 480 < DIM) {
        q16 = d_query[b_id * DIM + lane_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (lane_id + 512 < DIM) {
        q17 = d_query[b_id * DIM + lane_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (lane_id + 544 < DIM) {
        q18 = d_query[b_id * DIM + lane_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (lane_id + 576 < DIM) {
        q19 = d_query[b_id * DIM + lane_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (lane_id + 608 < DIM) {
        q20 = d_query[b_id * DIM + lane_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (lane_id + 640 < DIM) {
        q21 = d_query[b_id * DIM + lane_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (lane_id + 672 < DIM) {
        q22 = d_query[b_id * DIM + lane_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (lane_id + 704 < DIM) {
        q23 = d_query[b_id * DIM + lane_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (lane_id + 736 < DIM) {
        q24 = d_query[b_id * DIM + lane_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (lane_id + 768 < DIM) {
        q25 = d_query[b_id * DIM + lane_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (lane_id + 800 < DIM) {
        q26 = d_query[b_id * DIM + lane_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (lane_id + 832 < DIM) {
        q27 = d_query[b_id * DIM + lane_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (lane_id + 864 < DIM) {
        q28 = d_query[b_id * DIM + lane_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (lane_id + 896 < DIM) {
        q29 = d_query[b_id * DIM + lane_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (lane_id + 224 < DIM) {
        q30 = d_query[b_id * DIM + lane_id + 928];
    }
#endif
    
//Search    
    //First iteration
    // int cluster_id = d_enter_cluster[b_id];
    // int enter_points_num = min(d_rvq_indices_size[cluster_id], 2 * num_of_visited_points_one_batch);
    int enter_points_num = 2 * length_of_visited_points_one_batch;
    // int* enter_points_pos = d_rvq_indices[cluster_id];
    
    int iteration;
    //int enter_points_num = NumThreadQ * size_of_warp;
    int step_id;
    int substep_id;
    
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates /*+ num_of_visited_points_one_batch*/ + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;

        if (unrollt_id < num_of_candidates /*+ num_of_visited_points_one_batch*/) {

            neighbors_array[unrollt_id].first = MAX;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }
    __syncthreads();
    iteration = (enter_points_num + length_of_visited_points_one_batch -1) / length_of_visited_points_one_batch;
    for(int iter = 0; iter < iteration; iter++){

        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();

        for (int i = 0; i < length_of_visited_points_one_batch ; i++) {
            if (iter * length_of_visited_points_one_batch + i < enter_points_num){
               int block_id = iter * length_of_visited_points_one_batch + i;
               if(!test(block_id, random_number, data)){
                    int start_point_id = block_id * int(num_elements_per_block);
                    for(int l = 0; l < (num_elements_per_block + size_of_block - 1) / size_of_block; l++){
                        int unrollt_id = t_id + l * size_of_block;
                        if(unrollt_id < num_elements_per_block){
                            neighbors_array[num_of_candidates + i * length_of_block + unrollt_id].second = start_point_id + unrollt_id;
                        }
                    }
                    add(block_id, random_number, data);
                }
            }
            __syncthreads();
        }
        __syncthreads();
        
        
        for (int i = warp_id; i < num_of_visited_points_one_batch; i+=2) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);
            
            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = float(d_data[target_point_id * DIM + lane_id]);
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = float(d_data[target_point_id * DIM + lane_id + 32]);
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = float(d_data[target_point_id * DIM + lane_id + 64]);
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = float(d_data[target_point_id * DIM + lane_id + 96]);
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float(d_data[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float(d_data[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float(d_data[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float(d_data[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float(d_data[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float(d_data[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float(d_data[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float(d_data[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float(d_data[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float(d_data[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float(d_data[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float(d_data[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float(d_data[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float(d_data[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float(d_data[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float(d_data[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float(d_data[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float(d_data[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float(d_data[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float(d_data[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float(d_data[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float(d_data[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float(d_data[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float(d_data[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float(d_data[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float(d_data[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif
            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;
            }

        }
        
        __syncthreads();
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block-1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if ((neighbors_array[unrollt_id].first) > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                    
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        
    }
    __syncthreads();
    
    int flag_all_blocks = 1;
    int crt_flag = 0;
    int tmp_flag = (1 << min(min(num_of_explored_points, size_of_warp), enter_points_num)) - 1;;

    int tmp_search_width;
    int first_position_of_flag;
    int check_zero = 0;
    int hash_iteration = 0;
    iteration = 0;
    int iter = 0;
    int max_iter = num_of_explored_points / search_width;
    while (flag_all_blocks && iter < max_iter){
        iter++;
        
        for(int i = 0 ; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++){
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();
        tmp_search_width = 0;
        while(tmp_search_width < search_width && tmp_flag != 0){

            int first_position_of_tmp_flag = __ffs(tmp_flag) - 1;
            int neighbor_loc = iteration * size_of_warp + first_position_of_tmp_flag;
            
            first_position_of_flag = abs(neighbors_array[neighbor_loc].second);
            
            
            if(t_id == 0){
                neighbors_array[neighbor_loc].second = -neighbors_array[neighbor_loc].second;
            }
            tmp_flag &= ~(1 << first_position_of_tmp_flag);

            //读取邻居
            auto offset = size_t(first_position_of_flag) << offset_shift;
            int* neighbors_of_first_point = d_graph + offset;
            for (int i = 0; i < degree_of_point; i++) {
                int target_point = neighbors_of_first_point[i]; 
                if(target_point < total_num_of_points){
                    int block_id = target_point / int(num_elements_per_block);
                    if(!test(block_id, random_number, data)){
                        int start_point_id = block_id * int(num_elements_per_block);
                        for(int l = 0; l < (num_elements_per_block + size_of_block - 1) / size_of_block; l++){
                            int unrollt_id = t_id + size_of_block * l;
                            if(unrollt_id < num_elements_per_block){
                                neighbors_array[num_of_candidates + unrollt_id + tmp_search_width * degree_of_point * length_of_block + i * length_of_block].second = start_point_id + unrollt_id;
                            } 
                        }
                        add(block_id, random_number, data);
                    }
                }
                __syncthreads();
            }
            __syncthreads();

            
            tmp_search_width++;
            if(tmp_search_width == search_width) break;

            while(tmp_flag == 0 && iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp){
                iteration++;
                int unrollt_id = lane_id + size_of_warp * iteration;
                crt_flag = 0;
                if(unrollt_id < num_of_explored_points){
                    if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                        crt_flag = 1;
                    }else if(neighbors_array[unrollt_id].second == 0){
                        if(check_zero == 0){
                            check_zero = 1;
                            crt_flag = 1;
                        }
                    }
                }
                tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
            }
            
        }
        __syncthreads();
        if(tmp_search_width == 0) break;
        //计算query与search_width个点的邻居的距离
            

        for (int i = warp_id; i < tmp_search_width * degree_of_point * length_of_block; i+=2) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);

            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            
            
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = float(d_data[target_point_id * DIM + lane_id]);
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = float(d_data[target_point_id * DIM + lane_id + 32]);
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = float(d_data[target_point_id * DIM + lane_id + 64]);
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = float(d_data[target_point_id * DIM + lane_id + 96]);
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float(d_data[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float(d_data[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float(d_data[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float(d_data[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float(d_data[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float(d_data[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float(d_data[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float(d_data[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float(d_data[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float(d_data[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float(d_data[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float(d_data[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float(d_data[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float(d_data[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float(d_data[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float(d_data[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float(d_data[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float(d_data[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float(d_data[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float(d_data[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float(d_data[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float(d_data[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float(d_data[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float(d_data[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float(d_data[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float(d_data[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif

            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;    
            }

        }
        __syncthreads();
        // 对邻居列表排序
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        //将候选列表与邻居列表merge之后排序
        
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if(neighbors_array[unrollt_id + num_of_visited_points_one_batch].first == MAX) continue;
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        // //判断循环条件

        for (iteration = 0; iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp; iteration++) {
            int unrollt_id = lane_id + size_of_warp * iteration;
            crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                //crt_flag = flags[unrollt_id];
                if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                    crt_flag = 1;
                }else if(neighbors_array[unrollt_id].second == 0){
                    if(check_zero == 0){
                        check_zero = 1;
                        crt_flag = 1;
                    }
                }
            }
            tmp_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(tmp_flag != 0){
                break;
            }else if(iteration == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }
        
        // if(hash_iteration == 4){
        //     for(int i = 0; i < (size32 + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < size32){
        //             data[unrollt_id] = 0;
        //         }
        //     }
        //     __syncthreads();
            
        //     for(int i = 0; i < (num_of_candidates * num_hash + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < num_of_candidates * num_hash){
        //             int index = abs(neighbors_array[unrollt_id / num_hash].second);
        //             if(index < total_num_of_points){
        //                set_bit(hash_(unrollt_id % num_hash,index,random_number),data); 
        //             }
        //         }
        //     }
        //     //__syncthreads();
        //     hash_iteration = 0;
        // }
       
        hash_iteration++;
    }
    
    for (int i = 0; i < (num_of_results + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;
    
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id].second = abs(neighbors_array[unrollt_id].second);
            crt_result[unrollt_id].first = neighbors_array[unrollt_id].first;
        }
    }
}


template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void cagra_bam(array_d_t<char>* d_data, float* d_query, int* d_result, int total_num_of_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int search_width,
                    int** d_rvq_indices, int* d_rvq_indices_size, size_t hash_size, uint32_t* d_hash_table,
                    size_t page_size, size_t num_elements_per_block, unsigned long long* time_breakdown, size_t* IO_count, int *d_starling_index){

    // constexpr int WARP_SIZE = 32;
	// constexpr int NumWarpQ = 32;
	// constexpr int NumThreadQ = 1;
    
    #define DIM 128
    int t_id = threadIdx.x;
    size_t b_id = size_t(blockIdx.x);
    int size_of_warp = 32;
    int size_of_block = blockDim.x;
    size_t lane_id = size_t(threadIdx.x) % size_t(size_of_warp);
    int warp_id = threadIdx.x / size_of_warp;
    int warp_size = size_of_block / size_of_warp;
    int* crt_result = d_result + b_id * num_of_results;
    // float* crt_dis = d_dis + b_id * num_of_results;
    unsigned long long* crt_time_breakdown = time_breakdown + b_id * 4;
    int degree_of_point = (1 << offset_shift);
    int num_of_visited_points_one_batch = (search_width << offset_shift);
    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch;
    }

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
    array_d_t<float>* vector_data = reinterpret_cast<array_d_t<float>*>(d_data);
    array_d_t<int>* neighbor_data = reinterpret_cast<array_d_t<int>*>(d_data);

    // __shared__ uint32_t data[size32];
    uint32_t* data = d_hash_table + b_id * hash_size;
    uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
    };

#if DIM > 0
	float q1 = 0;
	if (lane_id < DIM) {
		q1 = d_query[b_id * DIM + lane_id];
	}
#endif
#if DIM > 32
    float q2 = 0;
    if (lane_id + 32 < DIM) {
        q2 = d_query[b_id * DIM + lane_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (lane_id + 64 < DIM) {
    	q3 = d_query[b_id * DIM + lane_id + 64];
   	}
#endif
#if DIM > 96
    float q4 = 0;
    if (lane_id + 96 < DIM) {
    	q4 = d_query[b_id * DIM + lane_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (lane_id + 128 < DIM) {
        q5 = d_query[b_id * DIM + lane_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (lane_id + 160 < DIM) {
        q6 = d_query[b_id * DIM + lane_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (lane_id + 192 < DIM) {
        q7 = d_query[b_id * DIM + lane_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (lane_id + 224 < DIM) {
        q8 = d_query[b_id * DIM + lane_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (lane_id + 256 < DIM) {
        q9 = d_query[b_id * DIM + lane_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (lane_id + 288 < DIM) {
        q10 = d_query[b_id * DIM + lane_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (lane_id + 320 < DIM) {
        q11 = d_query[b_id * DIM + lane_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (lane_id + 352 < DIM) {
        q12 = d_query[b_id * DIM + lane_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (lane_id + 384 < DIM) {
        q13 = d_query[b_id * DIM + lane_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (lane_id + 416 < DIM) {
        q14 = d_query[b_id * DIM + lane_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (lane_id + 448 < DIM) {
        q15 = d_query[b_id * DIM + lane_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (lane_id + 480 < DIM) {
        q16 = d_query[b_id * DIM + lane_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (lane_id + 512 < DIM) {
        q17 = d_query[b_id * DIM + lane_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (lane_id + 544 < DIM) {
        q18 = d_query[b_id * DIM + lane_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (lane_id + 576 < DIM) {
        q19 = d_query[b_id * DIM + lane_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (lane_id + 608 < DIM) {
        q20 = d_query[b_id * DIM + lane_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (lane_id + 640 < DIM) {
        q21 = d_query[b_id * DIM + lane_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (lane_id + 672 < DIM) {
        q22 = d_query[b_id * DIM + lane_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (lane_id + 704 < DIM) {
        q23 = d_query[b_id * DIM + lane_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (lane_id + 736 < DIM) {
        q24 = d_query[b_id * DIM + lane_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (lane_id + 768 < DIM) {
        q25 = d_query[b_id * DIM + lane_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (lane_id + 800 < DIM) {
        q26 = d_query[b_id * DIM + lane_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (lane_id + 832 < DIM) {
        q27 = d_query[b_id * DIM + lane_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (lane_id + 864 < DIM) {
        q28 = d_query[b_id * DIM + lane_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (lane_id + 896 < DIM) {
        q29 = d_query[b_id * DIM + lane_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (lane_id + 224 < DIM) {
        q30 = d_query[b_id * DIM + lane_id + 928];
    }
#endif
    
//Search    
    //First iteration
    // int cluster_id = d_enter_cluster[b_id];
    // int enter_points_num = min(max(search_width * num_of_visited_points_one_batch, num_of_candidates), d_rvq_indices_size[b_id]);
    // int* enter_points_pos = d_rvq_indices[b_id];
    // float* enter_points_data = d_rvq_data[b_id];
    int enter_points_num =  search_width * num_of_visited_points_one_batch;
    int iteration;
  
    int step_id;
    int substep_id;
    
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;

        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {

            neighbors_array[unrollt_id].first = MAX;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }
    // for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
    //     int unrollt_id = t_id + size_of_block * i;

    //     if (unrollt_id < num_of_candidates) {
    //         neighbors_array[unrollt_id] = tmp_d_result[num_of_candidates * b_id + unrollt_id];
    //     }
    //     else if(unrollt_id < num_of_candidates + num_of_visited_points_one_batch){
    //         neighbors_array[unrollt_id].first = MAX;
    //         neighbors_array[unrollt_id].second = total_num_of_points;
    //     }
    // }
    __syncthreads();
    iteration = (enter_points_num + num_of_visited_points_one_batch -1) / num_of_visited_points_one_batch;
    
    for(int iter = 0; iter < iteration; iter++){
        auto load_neighbors_start = clock64();
        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
            int unrollt_id = t_id + size_of_block * i;

            if (unrollt_id < num_of_visited_points_one_batch && iter * num_of_visited_points_one_batch + unrollt_id < enter_points_num
                /*&& !test(enter_points_pos[iter * num_of_visited_points_one_batch + unrollt_id], random_number, data)*/) { //enter_points_num
                
                // neighbors_array[num_of_candidates + unrollt_id].second = enter_points_pos[iter * num_of_visited_points_one_batch + unrollt_id];
                
                neighbors_array[num_of_candidates + unrollt_id].second = iter * num_of_visited_points_one_batch + unrollt_id;
                
                add(neighbors_array[num_of_candidates + unrollt_id].second, random_number, data);
            }
            else if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
            }
        }
        __syncthreads();
        auto load_neighbors_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[0] += load_neighbors_end - load_neighbors_start;
        }
        
        for (int i = warp_id; i < num_of_visited_points_one_batch; i += warp_size) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);
            // size_t target_point_id = size_t(iter * num_of_visited_points_one_batch + i);
            
            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            size_t offset = (target_point_id / num_elements_per_block) * page_size / sizeof(float) + (target_point_id % num_elements_per_block) * (DIM + size_t(degree_of_point + 1));

            auto load_data_start = clock64();
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p1 = float(catch_data[resver_target_point_id * DIM + lane_id]);
    // }else{
    //     p1 = (*vector_data)[offset + lane_id];
    // }
    p1 = (*vector_data)[offset + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p2 = float(catch_data[resver_target_point_id * DIM + lane_id + 32]);
    // }else{
    //     p2 = (*vector_data)[offset + lane_id + size_t(32)];
    // }
    p2 = (*vector_data)[offset + lane_id + size_t(32)];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p3 = float(catch_data[resver_target_point_id * DIM + lane_id + 64]);
    // }else{
    //     p3 = (*vector_data)[offset + lane_id + size_t(64)];
    // }
    p3 = (*vector_data)[offset + lane_id + size_t(64)];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p4 = float(catch_data[resver_target_point_id * DIM + lane_id + 96]);
    // }else{
    //     p4 = (*vector_data)[offset + lane_id + 96];
    // }
    p4 = (*vector_data)[offset + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float((*d_data)[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float((*d_data)[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float((*d_data)[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float((*d_data)[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float((*d_data)[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float((*d_data)[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float((*d_data)[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float((*d_data)[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float((*d_data)[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float((*d_data)[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float((*d_data)[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float((*d_data)[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float((*d_data)[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float((*d_data)[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float((*d_data)[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float((*d_data)[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float((*d_data)[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float((*d_data)[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float((*d_data)[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float((*d_data)[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float((*d_data)[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float((*d_data)[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float((*d_data)[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float((*d_data)[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float((*d_data)[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float((*d_data)[target_point_id * DIM + lane_id + 928]);
}
#endif
auto load_data_end = clock64();
if(t_id == 0){
    crt_time_breakdown[1] += load_data_end - load_data_start;
}

auto compute_distance_start = clock64();
#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif
            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;
                // IO_count[b_id]++;
                // if(b_id == 0){
                //     printf("id: %lu, dis: %f \n", target_point_id, dist);
                // }
            }
            auto compute_distance_end = clock64();
            if(t_id == 0){
                crt_time_breakdown[2] += compute_distance_end - compute_distance_start;
            }
        }
        
        __syncthreads();
        step_id = 1;
        substep_id = 1;
        auto sort_start = clock64();
        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block-1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if ((neighbors_array[unrollt_id].first) > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                    
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        auto sort_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[3] += sort_end - sort_start;
        }
    }
    __syncthreads();
    
    int flag_all_blocks = 1;
    int crt_flag = 0;
    int tmp_flag = (1 << min(min(num_of_explored_points, size_of_warp), enter_points_num)) - 1;

    int tmp_search_width;
    int first_position_of_flag;
    int check_zero = 0;
    int hash_iteration = 0;
    iteration = 0;
    // int iter = 0;
    // int max_iter = num_of_explored_points;
    while (flag_all_blocks /*&& iter < max_iter*/){
        // iter++;
        if(t_id == 0){
            IO_count[b_id]++;
        }
        for(int i = 0 ; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++){
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();
        tmp_search_width = 0;
        auto load_neighbors_start = clock64();
        while(tmp_search_width < search_width && tmp_flag != 0){

            int first_position_of_tmp_flag = __ffs(tmp_flag) - 1;
            int neighbor_loc = iteration * size_of_warp + first_position_of_tmp_flag;
            
            first_position_of_flag = abs(neighbors_array[neighbor_loc].second);
            if(t_id == 0){
                neighbors_array[neighbor_loc].second = -neighbors_array[neighbor_loc].second;
                // IO_count[b_id]++;
            }
            tmp_flag &= ~(1 << first_position_of_tmp_flag);

            //读取邻居
            // auto offset = size_t(first_position_of_flag) << offset_shift;
            size_t offset = (size_t(first_position_of_flag) / num_elements_per_block) * page_size / sizeof(int) + 
                            (size_t(first_position_of_flag) % num_elements_per_block) * (DIM + size_t(degree_of_point + 1)) + DIM;
            int tmp_degree_of_point = (*neighbor_data)[offset];
            // int* neighbors_of_first_point = d_graph + offset;
            // int* neighbors_of_first_point = &((*d_graph)[offset]);
            // int t = ((*d_graph)[offset]);
            // int* neighbors_of_first_point = &t;
            // if(first_position_of_flag == 0 && t_id == 0){
            //     for(int k = 0; k < degree_of_point; k++){
            //         printf("%d", (*d_graph)[offset + k]);
            //     }
            //     printf("\n");
            // }
            for (int i = 0; i < (degree_of_point + size_of_block - 1) / size_of_block; i++) {
                int unrollt_id = t_id + size_of_block * i;
                if (unrollt_id < degree_of_point) {
                    int neighbor_loc = num_of_candidates + unrollt_id + tmp_search_width * degree_of_point;
                    if(unrollt_id >= tmp_degree_of_point){
                        neighbors_array[neighbor_loc].second = total_num_of_points;
                    }else{
                        // int target_point = (*neighbor_data)[offset + size_t(unrollt_id + 1)]; 
                        int target_point = d_starling_index[(*neighbor_data)[offset + size_t(unrollt_id + 1)]]; 
                        if(test(target_point, random_number, data)){
                            neighbors_array[neighbor_loc].second = total_num_of_points;
                        }else{
                            // (*vector_data)[(target_point / num_elements_per_block) * page_size / sizeof(float)];
                            neighbors_array[neighbor_loc].second = target_point; 
                            add(target_point, random_number, data);
                        }
                    }
                    
                }
            }
            __syncthreads();

            
            tmp_search_width++;
            if(tmp_search_width == search_width) break;

            while(tmp_flag == 0 && iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp){
                iteration++;
                int unrollt_id = lane_id + size_of_warp * iteration;
                crt_flag = 0;
                if(unrollt_id < num_of_explored_points){
                    if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                        crt_flag = 1;
                    }else if(neighbors_array[unrollt_id].second == 0){
                        if(check_zero == 0){
                            check_zero = 1;
                            crt_flag = 1;
                        }
                    }
                }
                tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
            }
            
        }
        __syncthreads();
        auto load_neighbors_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[0] += load_neighbors_end - load_neighbors_start;
        }
        if(tmp_search_width == 0) break;
        //计算query与search_width个点的邻居的距离
    
        for (int i = warp_id; i < tmp_search_width * degree_of_point; i += warp_size) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);

            if (target_point_id >= total_num_of_points) {
                neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            // if(lane_id == 0 && i + warp_size < tmp_search_width * degree_of_point){
            //     size_t tmp_target_point_id = size_t(neighbors_array[num_of_candidates + i + warp_size].second);

            //     if (tmp_target_point_id >= total_num_of_points) {
            //         continue;
            //     }
            //     size_t offset = (tmp_target_point_id / num_elements_per_block) * page_size / sizeof(float);
            //     float tmp_data = (*vector_data)[offset];
            // }    

            size_t offset = (target_point_id / num_elements_per_block) * page_size / sizeof(float) + (target_point_id % num_elements_per_block) * (DIM + size_t(degree_of_point + 1));

auto load_data_start = clock64();
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p1 = float(catch_data[resver_target_point_id * DIM + lane_id]);
    // }else{
    //     p1 = (*vector_data)[offset + lane_id];
    // }
    p1 = (*vector_data)[offset + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p2 = float(catch_data[resver_target_point_id * DIM + lane_id + 32]);
    // }else{
    //     p2 = (*vector_data)[offset + lane_id + size_t(32)];
    // }
    p2 = (*vector_data)[offset + lane_id + size_t(32)];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p3 = float(catch_data[resver_target_point_id * DIM + lane_id + 64]);
    // }else{
    //     p3 = (*vector_data)[offset + lane_id + size_t(64)];
    // }
    p3 = (*vector_data)[offset + lane_id + size_t(64)];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    // if(resver_target_point_id < max_point_id){
    //     p4 = float(catch_data[resver_target_point_id * DIM + lane_id + 96]);
    // }else{
    //     p4 = (*vector_data)[offset + lane_id + 96];
    // }
    p4 = (*vector_data)[offset + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float((*d_data)[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float((*d_data)[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float((*d_data)[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float((*d_data)[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float((*d_data)[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float((*d_data)[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float((*d_data)[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float((*d_data)[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float((*d_data)[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float((*d_data)[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float((*d_data)[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float((*d_data)[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float((*d_data)[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float((*d_data)[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float((*d_data)[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float((*d_data)[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float((*d_data)[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float((*d_data)[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float((*d_data)[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float((*d_data)[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float((*d_data)[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float((*d_data)[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float((*d_data)[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float((*d_data)[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float((*d_data)[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float((*d_data)[target_point_id * DIM + lane_id + 928]);
}
#endif
auto load_data_end = clock64();
if(t_id == 0){
    crt_time_breakdown[1] += load_data_end - load_data_start;
}

auto compute_distance_start = clock64();
#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif

            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist; 
                // IO_count[b_id]++;  
            }
            auto compute_distance_end = clock64();
            if(t_id == 0){
                crt_time_breakdown[2] += compute_distance_end - compute_distance_start;
            }
        }
        __syncthreads();
        // 对邻居列表排序
        step_id = 1;
        substep_id = 1;
        auto sort_start = clock64();
        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        //将候选列表与邻居列表merge之后排序
        
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if(neighbors_array[unrollt_id + num_of_visited_points_one_batch].first == MAX) continue;
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        auto sort_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[3] += sort_end - sort_start;
        }
        // //判断循环条件
        auto get_next_point_start = clock64();
        for (iteration = 0; iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp; iteration++) {
            int unrollt_id = lane_id + size_of_warp * iteration;
            crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                //crt_flag = flags[unrollt_id];
                if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                    crt_flag = 1;
                }else if(neighbors_array[unrollt_id].second == 0){
                    if(check_zero == 0){
                        check_zero = 1;
                        crt_flag = 1;
                    }
                }
            }
            tmp_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(tmp_flag != 0){
                break;
            }else if(iteration == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }
        auto get_next_point_end = clock64();
        if(t_id == 0){
            crt_time_breakdown[0] += get_next_point_end - get_next_point_start;
        }
        // if(hash_iteration == 4){
        //     for(int i = 0; i < (size32 + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < size32){
        //             data[unrollt_id] = 0;
        //         }
        //     }
        //     __syncthreads();
            
        //     for(int i = 0; i < (num_of_candidates * num_hash + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < num_of_candidates * num_hash){
        //             int index = abs(neighbors_array[unrollt_id / num_hash].second);
        //             if(index < total_num_of_points){
        //                set_bit(hash_(unrollt_id % num_hash,index,random_number),data); 
        //             }
        //         }
        //     }
        //     //__syncthreads();
        //     hash_iteration = 0;
        // }
       
        // hash_iteration++;
    }
    // if(b_id == 0 && t_id == 0){
    //     for(int i = 0; i < num_of_results; i++){
    //         printf("%d %f\n", neighbors_array[i].second, neighbors_array[i].first);
    //     }
    // }
    for (int i = 0; i < (num_of_results + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;
    
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id] = abs(neighbors_array[unrollt_id].second);
            // crt_dis[unrollt_id] = neighbors_array[unrollt_id].first;
            
        }
    }
}

template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void cagra_bam_block(array_d_t<char>* d_data, float* d_query, int* d_result, int total_num_of_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int search_width,
                    int* d_enter_cluster, int** d_rvq_indices, float** d_rvq_data, int* d_rvq_indices_size, size_t hash_size, uint32_t* d_hash_table,
                    size_t max_point_id, float* catch_data, KernelPair<float, int>* tmp_d_result, size_t page_size, size_t num_elements_per_block
                    , int* d_starling_resver_index, int length_of_block/*, float *d_dis*/){

    // constexpr int WARP_SIZE = 32;
	// constexpr int NumWarpQ = 32;
	// constexpr int NumThreadQ = 1;
    
    #define DIM 128
    int t_id = threadIdx.x;
    size_t b_id = size_t(blockIdx.x);
    int size_of_warp = 32;
    int size_of_block = blockDim.x;
    size_t lane_id = size_t(threadIdx.x) % size_t(size_of_warp);
    int warp_id = threadIdx.x / size_of_warp;
    int warp_size = size_of_block / size_of_warp;
    int* crt_result = d_result + b_id * num_of_results;
    // float* crt_dis = d_dis + b_id * num_of_results;
    int degree_of_point = (1 << offset_shift);
    int num_of_visited_points_one_batch = (search_width << offset_shift) * length_of_block;
    int length_of_visited_points_one_batch = (search_width << offset_shift);
    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch;
    }

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
    array_d_t<float>* vector_data = reinterpret_cast<array_d_t<float>*>(d_data);
    array_d_t<int>* neighbor_data = reinterpret_cast<array_d_t<int>*>(d_data);

    // __shared__ uint32_t data[size32];
    uint32_t* data = d_hash_table + b_id * hash_size;
    uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
    };

#if DIM > 0
	float q1 = 0;
	if (lane_id < DIM) {
		q1 = d_query[b_id * DIM + lane_id];
	}
#endif
#if DIM > 32
    float q2 = 0;
    if (lane_id + 32 < DIM) {
        q2 = d_query[b_id * DIM + lane_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (lane_id + 64 < DIM) {
    	q3 = d_query[b_id * DIM + lane_id + 64];
   	}
#endif
#if DIM > 96
    float q4 = 0;
    if (lane_id + 96 < DIM) {
    	q4 = d_query[b_id * DIM + lane_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (lane_id + 128 < DIM) {
        q5 = d_query[b_id * DIM + lane_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (lane_id + 160 < DIM) {
        q6 = d_query[b_id * DIM + lane_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (lane_id + 192 < DIM) {
        q7 = d_query[b_id * DIM + lane_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (lane_id + 224 < DIM) {
        q8 = d_query[b_id * DIM + lane_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (lane_id + 256 < DIM) {
        q9 = d_query[b_id * DIM + lane_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (lane_id + 288 < DIM) {
        q10 = d_query[b_id * DIM + lane_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (lane_id + 320 < DIM) {
        q11 = d_query[b_id * DIM + lane_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (lane_id + 352 < DIM) {
        q12 = d_query[b_id * DIM + lane_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (lane_id + 384 < DIM) {
        q13 = d_query[b_id * DIM + lane_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (lane_id + 416 < DIM) {
        q14 = d_query[b_id * DIM + lane_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (lane_id + 448 < DIM) {
        q15 = d_query[b_id * DIM + lane_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (lane_id + 480 < DIM) {
        q16 = d_query[b_id * DIM + lane_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (lane_id + 512 < DIM) {
        q17 = d_query[b_id * DIM + lane_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (lane_id + 544 < DIM) {
        q18 = d_query[b_id * DIM + lane_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (lane_id + 576 < DIM) {
        q19 = d_query[b_id * DIM + lane_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (lane_id + 608 < DIM) {
        q20 = d_query[b_id * DIM + lane_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (lane_id + 640 < DIM) {
        q21 = d_query[b_id * DIM + lane_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (lane_id + 672 < DIM) {
        q22 = d_query[b_id * DIM + lane_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (lane_id + 704 < DIM) {
        q23 = d_query[b_id * DIM + lane_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (lane_id + 736 < DIM) {
        q24 = d_query[b_id * DIM + lane_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (lane_id + 768 < DIM) {
        q25 = d_query[b_id * DIM + lane_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (lane_id + 800 < DIM) {
        q26 = d_query[b_id * DIM + lane_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (lane_id + 832 < DIM) {
        q27 = d_query[b_id * DIM + lane_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (lane_id + 864 < DIM) {
        q28 = d_query[b_id * DIM + lane_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (lane_id + 896 < DIM) {
        q29 = d_query[b_id * DIM + lane_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (lane_id + 224 < DIM) {
        q30 = d_query[b_id * DIM + lane_id + 928];
    }
#endif
    
//Search    
    //First iteration
    int enter_points_num = min(max(length_of_visited_points_one_batch, num_of_candidates), d_rvq_indices_size[b_id]);
    int* enter_points_pos = d_rvq_indices[b_id];
    // int enter_points_num = 0;
    
    int iteration;
  
    int step_id;
    int substep_id;
    
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;

        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {

            neighbors_array[unrollt_id].first = MAX;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }
    // for (int i = 0; i < (num_of_candidates + size_of_block - 1) / size_of_block; i++) {
    //     int unrollt_id = t_id + size_of_block * i;

    //     if (unrollt_id < num_of_candidates) {
    //         neighbors_array[unrollt_id] = tmp_d_result[num_of_candidates * b_id + unrollt_id];
    //     }
        
    // }
    __syncthreads();

//     for (int i = 0; i < length_of_visited_points_one_batch ; i++) {
//         int block_id = tmp_d_result[b_id * length_of_visited_points_one_batch + i].second / int(num_elements_per_block);
//         if(!test(block_id, random_number, data)){
//             int start_point_id = block_id * int(num_elements_per_block);
//             for(int l = 0; l < (num_elements_per_block + size_of_block - 1) / size_of_block; l++){
//                 int unrollt_id = t_id + l * size_of_block;
//                 if(unrollt_id < num_elements_per_block){
//                     neighbors_array[num_of_candidates + i * length_of_block + unrollt_id].second = start_point_id + unrollt_id;
//                 }
//             }
//             add(block_id, random_number, data);
//         }
//         __syncthreads();
//     }
//     __syncthreads();
    
//     for (int i = warp_id; i < num_of_visited_points_one_batch; i += warp_size) {
//         size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);
//         // size_t target_point_id = size_t(iter * num_of_visited_points_one_batch + i);
        
//         if (target_point_id >= total_num_of_points) {
//             // neighbors_array[num_of_candidates + i].first = MAX;
//             continue;
//         }
//         size_t resver_target_point_id = size_t(d_starling_resver_index[target_point_id]);
//         size_t offset = (target_point_id / num_elements_per_block) * page_size / sizeof(float) + (target_point_id % num_elements_per_block) * (DIM + size_t(degree_of_point + 1));

// #if DIM > 0
// float p1 = 0;
// if (lane_id < DIM) {
// if(resver_target_point_id < max_point_id){
//     p1 = float(catch_data[resver_target_point_id * DIM + lane_id]);
// }else{
//     p1 = (*vector_data)[offset + lane_id];
// }
// // p1 = (*vector_data)[offset + lane_id];
// }
// #endif
// #if DIM > 32
// float p2 = 0;
// if (lane_id + 32 < DIM) {
// if(resver_target_point_id < max_point_id){
//     p2 = float(catch_data[resver_target_point_id * DIM + lane_id + 32]);
// }else{
//     p2 = (*vector_data)[offset + lane_id + size_t(32)];
// }
// // p2 = (*vector_data)[offset + lane_id + size_t(32)];
// }
// #endif
// #if DIM > 64
// float p3 = 0;
// if (lane_id + 64 < DIM) {
// if(resver_target_point_id < max_point_id){
//     p3 = float(catch_data[resver_target_point_id * DIM + lane_id + 64]);
// }else{
//     p3 = (*vector_data)[offset + lane_id + size_t(64)];
// }
// // p3 = (*vector_data)[offset + lane_id + size_t(64)];
// }
// #endif
// #if DIM > 96
// float p4 = 0;
// if (lane_id + 96 < DIM) {
// if(resver_target_point_id < max_point_id){
//     p4 = float(catch_data[resver_target_point_id * DIM + lane_id + 96]);
// }else{
//     p4 = (*vector_data)[offset + lane_id + 96];
// }
// // p4 = (*vector_data)[offset + lane_id + 96];
// }
// #endif
// #if DIM > 128
// float p5 = 0;
// if (lane_id + 128 < DIM) {
// p5 = float((*d_data)[target_point_id * DIM + lane_id + 128]);
// }
// #endif
// #if DIM > 160
// float p6 = 0;
// if (lane_id + 160 < DIM) {
// p6 = float((*d_data)[target_point_id * DIM + lane_id + 160]);
// }
// #endif
// #if DIM > 192
// float p7 = 0;
// if (lane_id + 192 < DIM) {
// p7 = float((*d_data)[target_point_id * DIM + lane_id + 192]);
// }
// #endif
// #if DIM > 224
// float p8 = 0;
// if (lane_id + 224 < DIM) {
// p8 = float((*d_data)[target_point_id * DIM + lane_id + 224]);
// }
// #endif
// #if DIM > 256
// float p9 = 0;
// if (lane_id + 256 < DIM) {
// p9 = float((*d_data)[target_point_id * DIM + lane_id + 256]);
// }
// #endif
// #if DIM > 288
// float p10 = 0;
// if (lane_id + 288 < DIM) {
// p10 = float((*d_data)[target_point_id * DIM + lane_id + 288]);
// }
// #endif
// #if DIM > 320
// float p11 = 0;
// if (lane_id + 320 < DIM) {
// p11 = float((*d_data)[target_point_id * DIM + lane_id + 320]);
// }
// #endif
// #if DIM > 352
// float p12 = 0;
// if (lane_id + 352 < DIM) {
// p12 = float((*d_data)[target_point_id * DIM + lane_id + 352]);
// }
// #endif
// #if DIM > 384
// float p13 = 0;
// if (lane_id + 384 < DIM) {
// p13 = float((*d_data)[target_point_id * DIM + lane_id + 384]);
// }
// #endif
// #if DIM > 416
// float p14 = 0;
// if (lane_id + 416 < DIM) {
// p14 = float((*d_data)[target_point_id * DIM + lane_id + 416]);
// }
// #endif
// #if DIM > 448
// float p15 = 0;
// if (lane_id + 448 < DIM) {
// p15 = float((*d_data)[target_point_id * DIM + lane_id + 448]);
// }
// #endif
// #if DIM > 480
// float p16 = 0;
// if (lane_id + 480 < DIM) {
// p16 = float((*d_data)[target_point_id * DIM + lane_id + 480]);
// }
// #endif
// #if DIM > 512
// float p17 = 0;
// if (lane_id + 512 < DIM) {
// p17 = float((*d_data)[target_point_id * DIM + lane_id + 512]);
// }
// #endif
// #if DIM > 544
// float p18 = 0;
// if (lane_id + 544 < DIM) {
// p18 = float((*d_data)[target_point_id * DIM + lane_id + 544]);
// }
// #endif
// #if DIM > 576
// float p19 = 0;
// if (lane_id + 576 < DIM) {
// p19 = float((*d_data)[target_point_id * DIM + lane_id + 576]);
// }
// #endif
// #if DIM > 608
// float p20 = 0;
// if (lane_id + 608 < DIM) {
// p20 = float((*d_data)[target_point_id * DIM + lane_id + 608]);
// }
// #endif
// #if DIM > 640
// float p21 = 0;
// if (lane_id + 640 < DIM) {
// p21 = float((*d_data)[target_point_id * DIM + lane_id + 640]);
// }
// #endif
// #if DIM > 672
// float p22 = 0;
// if (lane_id + 672 < DIM) {
// p22 = float((*d_data)[target_point_id * DIM + lane_id + 672]);
// }
// #endif
// #if DIM > 704
// float p23 = 0;
// if (lane_id + 704 < DIM) {
// p23 = float((*d_data)[target_point_id * DIM + lane_id + 704]);
// }
// #endif
// #if DIM > 736
// float p24 = 0;
// if (lane_id + 736 < DIM) {
// p24 = float((*d_data)[target_point_id * DIM + lane_id + 736]);
// }
// #endif
// #if DIM > 768
// float p25 = 0;
// if (lane_id + 768 < DIM) {
// p25 = float((*d_data)[target_point_id * DIM + lane_id + 768]);
// }
// #endif
// #if DIM > 800
// float p26 = 0;
// if (lane_id + 800 < DIM) {
// p26 = float((*d_data)[target_point_id * DIM + lane_id + 800]);
// }
// #endif
// #if DIM > 832
// float p27 = 0;
// if (lane_id + 832 < DIM) {
// p27 = float((*d_data)[target_point_id * DIM + lane_id + 832]);
// }
// #endif
// #if DIM > 864
// float p28 = 0;
// if (lane_id + 864 < DIM) {
// p28 = float((*d_data)[target_point_id * DIM + lane_id + 864]);
// }
// #endif
// #if DIM > 896
// float p29 = 0;
// if (lane_id + 896 < DIM) {
// p29 = float((*d_data)[target_point_id * DIM + lane_id + 896]);
// }
// #endif
// #if DIM > 928
// float p30 = 0;
// if (lane_id + 224 < DIM) {
// p30 = float((*d_data)[target_point_id * DIM + lane_id + 928]);
// }
// #endif


// #if USE_L2_DIST_
// #if DIM > 0
// float delta1 = (p1 - q1) * (p1 - q1);
// #endif
// #if DIM > 32
// float delta2 = (p2 - q2) * (p2 - q2);
// #endif
// #if DIM > 64
// float delta3 = (p3 - q3) * (p3 - q3);
// #endif
// #if DIM > 96
// float delta4 = (p4 - q4) * (p4 - q4);
// #endif
// #if DIM > 128
// float delta5 = (p5 - q5) * (p5 - q5);
// #endif
// #if DIM > 160
// float delta6 = (p6 - q6) * (p6 - q6);
// #endif
// #if DIM > 192
// float delta7 = (p7 - q7) * (p7 - q7);
// #endif
// #if DIM > 224
// float delta8 = (p8 - q8) * (p8 - q8);
// #endif
// #if DIM > 256
// float delta9 = (p9 - q9) * (p9 - q9);
// #endif
// #if DIM > 288
// float delta10 = (p10 - q10) * (p10 - q10);
// #endif
// #if DIM > 320
// float delta11 = (p11 - q11) * (p11 - q11);
// #endif
// #if DIM > 352
// float delta12 = (p12 - q12) * (p12 - q12);
// #endif
// #if DIM > 384
// float delta13 = (p13 - q13) * (p13 - q13);
// #endif
// #if DIM > 416
// float delta14 = (p14 - q14) * (p14 - q14);
// #endif
// #if DIM > 448
// float delta15 = (p15 - q15) * (p15 - q15);
// #endif
// #if DIM > 480
// float delta16 = (p16 - q16) * (p16 - q16);
// #endif
// #if DIM > 512
// float delta17 = (p17 - q17) * (p17 - q17);
// #endif
// #if DIM > 544
// float delta18 = (p18 - q18) * (p18 - q18);
// #endif
// #if DIM > 576
// float delta19 = (p19 - q19) * (p19 - q19);
// #endif
// #if DIM > 608
// float delta20 = (p20 - q20) * (p20 - q20);
// #endif
// #if DIM > 640
// float delta21 = (p21 - q21) * (p21 - q21);
// #endif
// #if DIM > 672
// float delta22 = (p22 - q22) * (p22 - q22);
// #endif
// #if DIM > 704
// float delta23 = (p23 - q23) * (p23 - q23);
// #endif
// #if DIM > 736
// float delta24 = (p24 - q24) * (p24 - q24);
// #endif
// #if DIM > 768
// float delta25 = (p25 - q25) * (p25 - q25);
// #endif
// #if DIM > 800
// float delta26 = (p26 - q26) * (p26 - q26);
// #endif
// #if DIM > 832
// float delta27 = (p27 - q27) * (p27 - q27);
// #endif
// #if DIM > 864
// float delta28 = (p28 - q28) * (p28 - q28);
// #endif
// #if DIM > 896
// float delta29 = (p29 - q29) * (p29 - q29);
// #endif
// #if DIM > 928
// float delta30 = (p30 - q30) * (p30 - q30);
// #endif
// #endif           
// #if USE_L2_DIST_
// float dist = 0;
// #if DIM > 0
// dist += delta1;
// #endif
// #if DIM > 32
// dist += delta2;
// #endif
// #if DIM > 64
// dist += delta3;
// #endif
// #if DIM > 96
// dist += delta4;
// #endif
// #if DIM > 128
// dist += delta5;
// #endif
// #if DIM > 160
// dist += delta6;
// #endif
// #if DIM > 192
// dist += delta7;
// #endif
// #if DIM > 224
// dist += delta8;
// #endif
// #if DIM > 256
// dist += delta9;
// #endif
// #if DIM > 288
// dist += delta10;
// #endif
// #if DIM > 320
// dist += delta11;
// #endif
// #if DIM > 352
// dist += delta12;
// #endif
// #if DIM > 384
// dist += delta13;
// #endif
// #if DIM > 416
// dist += delta14;
// #endif
// #if DIM > 448
// dist += delta15;
// #endif
// #if DIM > 480
// dist += delta16;
// #endif
// #if DIM > 512
// dist += delta17;
// #endif
// #if DIM > 544
// dist += delta18;
// #endif
// #if DIM > 576
// dist += delta19;
// #endif
// #if DIM > 608
// dist += delta20;
// #endif
// #if DIM > 640
// dist += delta21;
// #endif
// #if DIM > 672
// dist += delta22;
// #endif
// #if DIM > 704
// dist += delta23;
// #endif
// #if DIM > 736
// dist += delta24;
// #endif
// #if DIM > 768
// dist += delta25;
// #endif
// #if DIM > 800
// dist += delta26;
// #endif
// #if DIM > 832
// dist += delta27;
// #endif
// #if DIM > 864
// dist += delta28;
// #endif
// #if DIM > 896
// dist += delta29;
// #endif
// #if DIM > 928
// dist += delta30;
// #endif
// #endif
// #if USE_L2_DIST_
// dist += __shfl_down_sync(FULL_MASK, dist, 16);
// dist += __shfl_down_sync(FULL_MASK, dist, 8);
// dist += __shfl_down_sync(FULL_MASK, dist, 4);
// dist += __shfl_down_sync(FULL_MASK, dist, 2);
// dist += __shfl_down_sync(FULL_MASK, dist, 1);
// #endif

// #if USE_L2_DIST_
// //dist = sqrt(dist);
// #endif
//         if (lane_id == 0) {
//             neighbors_array[num_of_candidates + i].first = dist;
//             // if(b_id == 0){
//             //     printf("id: %lu, dis: %f \n", target_point_id, dist);
//             // }
//         }

//     }
    
//     __syncthreads();
//     step_id = 1;
//     substep_id = 1;

//     for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
//         substep_id = step_id;

//         for (; substep_id >= 1; substep_id /= 2) {
//             for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block-1) / size_of_block; temparory_id++) {
//                 int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                
//                 if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
//                     if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
//                         if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
//                             temporary_neighbor = neighbors_array[unrollt_id];
//                             neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
//                             neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
//                         }
//                     } else {
//                         if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
//                             temporary_neighbor = neighbors_array[unrollt_id];
//                             neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
//                             neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
//                         }
//                     }
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     __syncthreads();
//     for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
//         int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
//         if (unrollt_id < num_of_candidates) {
//             if ((neighbors_array[unrollt_id].first) > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
//                 temporary_neighbor = neighbors_array[unrollt_id];
//                 neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
//                 neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                
//             }
//         }
//     }
//     __syncthreads();
//     step_id = num_of_candidates / 2;
//     substep_id = num_of_candidates / 2;
//     for (; substep_id >= 1; substep_id /= 2) {
//         for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
//             int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
//             if (unrollt_id < num_of_candidates) {
//                 if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
//                     if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
//                         temporary_neighbor = neighbors_array[unrollt_id];
//                         neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
//                         neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
//                     }
//                 } else {
//                     if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
//                         temporary_neighbor = neighbors_array[unrollt_id];
//                         neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
//                         neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        
//                     }
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     __syncthreads();
    iteration = (enter_points_num + length_of_visited_points_one_batch -1) / length_of_visited_points_one_batch;
    
    for(int iter = 0; iter < iteration; iter++){

        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++) {
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();

        for (int i = 0; i < length_of_visited_points_one_batch ; i++) {
            if (iter * length_of_visited_points_one_batch + i < enter_points_num){
               int block_id = enter_points_pos[iter * length_of_visited_points_one_batch + i] / int(num_elements_per_block);
               if(!test(block_id, random_number, data)){
                    int start_point_id = block_id * int(num_elements_per_block);
                    for(int l = 0; l < (num_elements_per_block + size_of_block - 1) / size_of_block; l++){
                        int unrollt_id = t_id + l * size_of_block;
                        if(unrollt_id < num_elements_per_block){
                            neighbors_array[num_of_candidates + i * length_of_block + unrollt_id].second = start_point_id + unrollt_id;
                        }
                    }
                    add(block_id, random_number, data);
                }
            }
            __syncthreads();
        }
        __syncthreads();
        
        for (int i = warp_id; i < num_of_visited_points_one_batch; i += warp_size) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);
            // size_t target_point_id = size_t(iter * num_of_visited_points_one_batch + i);
            
            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            size_t resver_target_point_id = size_t(d_starling_resver_index[target_point_id]);
            size_t offset = (target_point_id / num_elements_per_block) * page_size / sizeof(float) + (target_point_id % num_elements_per_block) * (DIM + size_t(degree_of_point + 1));

#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    if(resver_target_point_id < max_point_id){
        p1 = float(catch_data[resver_target_point_id * DIM + lane_id]);
    }else{
        p1 = (*vector_data)[offset + lane_id];
    }
    // p1 = (*vector_data)[offset + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    if(resver_target_point_id < max_point_id){
        p2 = float(catch_data[resver_target_point_id * DIM + lane_id + 32]);
    }else{
        p2 = (*vector_data)[offset + lane_id + size_t(32)];
    }
    // p2 = (*vector_data)[offset + lane_id + size_t(32)];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    if(resver_target_point_id < max_point_id){
        p3 = float(catch_data[resver_target_point_id * DIM + lane_id + 64]);
    }else{
        p3 = (*vector_data)[offset + lane_id + size_t(64)];
    }
    // p3 = (*vector_data)[offset + lane_id + size_t(64)];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    if(resver_target_point_id < max_point_id){
        p4 = float(catch_data[resver_target_point_id * DIM + lane_id + 96]);
    }else{
        p4 = (*vector_data)[offset + lane_id + 96];
    }
    // p4 = (*vector_data)[offset + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float((*d_data)[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float((*d_data)[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float((*d_data)[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float((*d_data)[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float((*d_data)[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float((*d_data)[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float((*d_data)[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float((*d_data)[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float((*d_data)[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float((*d_data)[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float((*d_data)[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float((*d_data)[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float((*d_data)[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float((*d_data)[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float((*d_data)[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float((*d_data)[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float((*d_data)[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float((*d_data)[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float((*d_data)[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float((*d_data)[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float((*d_data)[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float((*d_data)[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float((*d_data)[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float((*d_data)[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float((*d_data)[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float((*d_data)[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif
            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;
                // if(b_id == 0){
                //     printf("id: %lu, dis: %f \n", target_point_id, dist);
                // }
            }

        }
        
        __syncthreads();
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block-1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if ((neighbors_array[unrollt_id].first) > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                    
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        
    }
    __syncthreads();
    
    int flag_all_blocks = 1;
    int crt_flag = 0;
    int tmp_flag = (1 << min(min(num_of_explored_points, size_of_warp), enter_points_num)) - 1;;

    int tmp_search_width;
    int first_position_of_flag;
    int check_zero = 0;
    int hash_iteration = 0;
    iteration = 0;
    // int iter = 0;
    // int max_iter = 1.5 * num_of_explored_points / search_width;
    while (flag_all_blocks /*&& iter < max_iter*/){
        // iter++;
        
        for(int i = 0 ; i < (num_of_visited_points_one_batch + size_of_block - 1) / size_of_block; i++){
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < num_of_visited_points_one_batch){
                neighbors_array[num_of_candidates + unrollt_id].first = MAX;
                neighbors_array[num_of_candidates + unrollt_id].second = total_num_of_points;
            }
        }
        __syncthreads();
        tmp_search_width = 0;
        while(tmp_search_width < search_width && tmp_flag != 0){

            int first_position_of_tmp_flag = __ffs(tmp_flag) - 1;
            int neighbor_loc = iteration * size_of_warp + first_position_of_tmp_flag;
            
            first_position_of_flag = abs(neighbors_array[neighbor_loc].second);
            if(t_id == 0){
                neighbors_array[neighbor_loc].second = -neighbors_array[neighbor_loc].second;
            }
            tmp_flag &= ~(1 << first_position_of_tmp_flag);

            //读取邻居
            // auto offset = size_t(first_position_of_flag) << offset_shift;
            size_t offset = (size_t(first_position_of_flag) / num_elements_per_block) * page_size / sizeof(int) + 
                            (size_t(first_position_of_flag) % num_elements_per_block) * (DIM + size_t(degree_of_point + 1)) + DIM;
            int tmp_degree_of_point = (*neighbor_data)[offset];
            // int* neighbors_of_first_point = d_graph + offset;
            // int* neighbors_of_first_point = &((*d_graph)[offset]);
            // int t = ((*d_graph)[offset]);
            // int* neighbors_of_first_point = &t;
            // if(first_position_of_flag == 0 && t_id == 0){
            //     for(int k = 0; k < degree_of_point; k++){
            //         printf("%d", (*d_graph)[offset + k]);
            //     }
            //     printf("\n");
            // }
            for (int i = 0; i < tmp_degree_of_point; i++) {
                int target_point = (*neighbor_data)[offset + size_t(i + 1)]; 
                int block_id = target_point / int(num_elements_per_block);
                if(!test(block_id, random_number, data)){
                    int start_point_id = block_id * int(num_elements_per_block);
                    for(int l = 0; l < (num_elements_per_block + size_of_block - 1) / size_of_block; l++){
                        int unrollt_id = t_id + size_of_block * l;
                        if(unrollt_id < num_elements_per_block){
                            neighbors_array[num_of_candidates + unrollt_id + tmp_search_width * degree_of_point * length_of_block + i * length_of_block].second = start_point_id + unrollt_id;
                        } 
                    }
                    add(block_id, random_number, data);
                }
                __syncthreads();
            }
            __syncthreads();

            
            tmp_search_width++;
            if(tmp_search_width == search_width) break;

            while(tmp_flag == 0 && iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp){
                iteration++;
                int unrollt_id = lane_id + size_of_warp * iteration;
                crt_flag = 0;
                if(unrollt_id < num_of_explored_points){
                    if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                        crt_flag = 1;
                    }else if(neighbors_array[unrollt_id].second == 0){
                        if(check_zero == 0){
                            check_zero = 1;
                            crt_flag = 1;
                        }
                    }
                }
                tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
            }
            
        }
        __syncthreads();
        if(tmp_search_width == 0) break;
        //计算query与search_width个点的邻居的距离
    
        for (int i = warp_id; i < tmp_search_width * degree_of_point * length_of_block; i += warp_size) {
            size_t target_point_id = size_t(neighbors_array[num_of_candidates + i].second);

            if (target_point_id >= total_num_of_points) {
                // neighbors_array[num_of_candidates + i].first = MAX;
                continue;
            }
            // if(lane_id == 0 && i + warp_size < tmp_search_width * degree_of_point){
            //     size_t tmp_target_point_id = size_t(neighbors_array[num_of_candidates + i + warp_size].second);

            //     if (tmp_target_point_id >= total_num_of_points) {
            //         continue;
            //     }
            //     size_t offset = (tmp_target_point_id / num_elements_per_block) * page_size / sizeof(float);
            //     float tmp_data = (*vector_data)[offset];
            // }    

            size_t resver_target_point_id = size_t(d_starling_resver_index[target_point_id]);
            size_t offset = (target_point_id / num_elements_per_block) * page_size / sizeof(float) + (target_point_id % num_elements_per_block) * (DIM + size_t(degree_of_point + 1));

#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    if(resver_target_point_id < max_point_id){
        p1 = float(catch_data[resver_target_point_id * DIM + lane_id]);
    }else{
        p1 = (*vector_data)[offset + lane_id];
    }
    // p1 = (*vector_data)[offset + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    if(resver_target_point_id < max_point_id){
        p2 = float(catch_data[resver_target_point_id * DIM + lane_id + 32]);
    }else{
        p2 = (*vector_data)[offset + lane_id + size_t(32)];
    }
    // p2 = (*vector_data)[offset + lane_id + size_t(32)];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    if(resver_target_point_id < max_point_id){
        p3 = float(catch_data[resver_target_point_id * DIM + lane_id + 64]);
    }else{
        p3 = (*vector_data)[offset + lane_id + size_t(64)];
    }
    // p3 = (*vector_data)[offset + lane_id + size_t(64)];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    if(resver_target_point_id < max_point_id){
        p4 = float(catch_data[resver_target_point_id * DIM + lane_id + 96]);
    }else{
        p4 = (*vector_data)[offset + lane_id + 96];
    }
    // p4 = (*vector_data)[offset + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = float((*d_data)[target_point_id * DIM + lane_id + 128]);
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = float((*d_data)[target_point_id * DIM + lane_id + 160]);
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = float((*d_data)[target_point_id * DIM + lane_id + 192]);
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = float((*d_data)[target_point_id * DIM + lane_id + 224]);
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = float((*d_data)[target_point_id * DIM + lane_id + 256]);
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = float((*d_data)[target_point_id * DIM + lane_id + 288]);
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = float((*d_data)[target_point_id * DIM + lane_id + 320]);
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = float((*d_data)[target_point_id * DIM + lane_id + 352]);
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = float((*d_data)[target_point_id * DIM + lane_id + 384]);
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = float((*d_data)[target_point_id * DIM + lane_id + 416]);
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = float((*d_data)[target_point_id * DIM + lane_id + 448]);
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = float((*d_data)[target_point_id * DIM + lane_id + 480]);
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = float((*d_data)[target_point_id * DIM + lane_id + 512]);
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = float((*d_data)[target_point_id * DIM + lane_id + 544]);
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = float((*d_data)[target_point_id * DIM + lane_id + 576]);
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = float((*d_data)[target_point_id * DIM + lane_id + 608]);
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = float((*d_data)[target_point_id * DIM + lane_id + 640]);
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = float((*d_data)[target_point_id * DIM + lane_id + 672]);
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = float((*d_data)[target_point_id * DIM + lane_id + 704]);
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = float((*d_data)[target_point_id * DIM + lane_id + 736]);
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = float((*d_data)[target_point_id * DIM + lane_id + 768]);
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = float((*d_data)[target_point_id * DIM + lane_id + 800]);
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = float((*d_data)[target_point_id * DIM + lane_id + 832]);
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = float((*d_data)[target_point_id * DIM + lane_id + 864]);
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = float((*d_data)[target_point_id * DIM + lane_id + 896]);
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = float((*d_data)[target_point_id * DIM + lane_id + 928]);
}
#endif


#if USE_L2_DIST_
#if DIM > 0
    float delta1 = (p1 - q1) * (p1 - q1);
#endif
#if DIM > 32
    float delta2 = (p2 - q2) * (p2 - q2);
#endif
#if DIM > 64
    float delta3 = (p3 - q3) * (p3 - q3);
#endif
#if DIM > 96
    float delta4 = (p4 - q4) * (p4 - q4);
#endif
#if DIM > 128
    float delta5 = (p5 - q5) * (p5 - q5);
#endif
#if DIM > 160
    float delta6 = (p6 - q6) * (p6 - q6);
#endif
#if DIM > 192
    float delta7 = (p7 - q7) * (p7 - q7);
#endif
#if DIM > 224
    float delta8 = (p8 - q8) * (p8 - q8);
#endif
#if DIM > 256
    float delta9 = (p9 - q9) * (p9 - q9);
#endif
#if DIM > 288
    float delta10 = (p10 - q10) * (p10 - q10);
#endif
#if DIM > 320
    float delta11 = (p11 - q11) * (p11 - q11);
#endif
#if DIM > 352
    float delta12 = (p12 - q12) * (p12 - q12);
#endif
#if DIM > 384
    float delta13 = (p13 - q13) * (p13 - q13);
#endif
#if DIM > 416
    float delta14 = (p14 - q14) * (p14 - q14);
#endif
#if DIM > 448
    float delta15 = (p15 - q15) * (p15 - q15);
#endif
#if DIM > 480
    float delta16 = (p16 - q16) * (p16 - q16);
#endif
#if DIM > 512
    float delta17 = (p17 - q17) * (p17 - q17);
#endif
#if DIM > 544
    float delta18 = (p18 - q18) * (p18 - q18);
#endif
#if DIM > 576
    float delta19 = (p19 - q19) * (p19 - q19);
#endif
#if DIM > 608
    float delta20 = (p20 - q20) * (p20 - q20);
#endif
#if DIM > 640
    float delta21 = (p21 - q21) * (p21 - q21);
#endif
#if DIM > 672
    float delta22 = (p22 - q22) * (p22 - q22);
#endif
#if DIM > 704
    float delta23 = (p23 - q23) * (p23 - q23);
#endif
#if DIM > 736
    float delta24 = (p24 - q24) * (p24 - q24);
#endif
#if DIM > 768
    float delta25 = (p25 - q25) * (p25 - q25);
#endif
#if DIM > 800
    float delta26 = (p26 - q26) * (p26 - q26);
#endif
#if DIM > 832
    float delta27 = (p27 - q27) * (p27 - q27);
#endif
#if DIM > 864
    float delta28 = (p28 - q28) * (p28 - q28);
#endif
#if DIM > 896
    float delta29 = (p29 - q29) * (p29 - q29);
#endif
#if DIM > 928
    float delta30 = (p30 - q30) * (p30 - q30);
#endif
#endif           
#if USE_L2_DIST_
    float dist = 0;
#if DIM > 0
    dist += delta1;
#endif
#if DIM > 32
    dist += delta2;
#endif
#if DIM > 64
    dist += delta3;
#endif
#if DIM > 96
    dist += delta4;
#endif
#if DIM > 128
    dist += delta5;
#endif
#if DIM > 160
    dist += delta6;
#endif
#if DIM > 192
    dist += delta7;
#endif
#if DIM > 224
    dist += delta8;
#endif
#if DIM > 256
    dist += delta9;
#endif
#if DIM > 288
    dist += delta10;
#endif
#if DIM > 320
    dist += delta11;
#endif
#if DIM > 352
    dist += delta12;
#endif
#if DIM > 384
    dist += delta13;
#endif
#if DIM > 416
    dist += delta14;
#endif
#if DIM > 448
    dist += delta15;
#endif
#if DIM > 480
    dist += delta16;
#endif
#if DIM > 512
    dist += delta17;
#endif
#if DIM > 544
    dist += delta18;
#endif
#if DIM > 576
    dist += delta19;
#endif
#if DIM > 608
    dist += delta20;
#endif
#if DIM > 640
    dist += delta21;
#endif
#if DIM > 672
    dist += delta22;
#endif
#if DIM > 704
    dist += delta23;
#endif
#if DIM > 736
    dist += delta24;
#endif
#if DIM > 768
    dist += delta25;
#endif
#if DIM > 800
    dist += delta26;
#endif
#if DIM > 832
    dist += delta27;
#endif
#if DIM > 864
    dist += delta28;
#endif
#if DIM > 896
    dist += delta29;
#endif
#if DIM > 928
    dist += delta30;
#endif
#endif
#if USE_L2_DIST_
dist += __shfl_down_sync(FULL_MASK, dist, 16);
dist += __shfl_down_sync(FULL_MASK, dist, 8);
dist += __shfl_down_sync(FULL_MASK, dist, 4);
dist += __shfl_down_sync(FULL_MASK, dist, 2);
dist += __shfl_down_sync(FULL_MASK, dist, 1);
#endif

#if USE_L2_DIST_
//dist = sqrt(dist);
#endif

            if (lane_id == 0) {
                neighbors_array[num_of_candidates + i].first = dist;   
            }

        }
        __syncthreads();
        // 对邻居列表排序
        step_id = 1;
        substep_id = 1;

        for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
            substep_id = step_id;

            for (; substep_id >= 1; substep_id /= 2) {
                for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                    int unrollt_id = num_of_candidates + ((t_id + size_of_block * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                    
                    if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                        if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                            if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        } else {
                            if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                                temporary_neighbor = neighbors_array[unrollt_id];
                                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                                neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        //将候选列表与邻居列表merge之后排序
        
        for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_block - 1) / size_of_block; temparory_id++) {
            int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_block * temparory_id;
            if (unrollt_id < num_of_candidates) {
                if(neighbors_array[unrollt_id + num_of_visited_points_one_batch].first == MAX) continue;
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
                    neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
                }
            }
        }
        __syncthreads();
        step_id = num_of_candidates / 2;
        substep_id = num_of_candidates / 2;
        for (; substep_id >= 1; substep_id /= 2) {
            for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_block - 1) / size_of_block; temparory_id++) {
                int unrollt_id = ((t_id + size_of_block * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_block * temparory_id) & (substep_id - 1));
                if (unrollt_id < num_of_candidates) {
                    if (((t_id + size_of_block * temparory_id) / step_id) % 2 == 0) {
                        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    } else {
                        if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                            temporary_neighbor = neighbors_array[unrollt_id];
                            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                            neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        // //判断循环条件

        for (iteration = 0; iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp; iteration++) {
            int unrollt_id = lane_id + size_of_warp * iteration;
            crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                //crt_flag = flags[unrollt_id];
                if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < total_num_of_points){
                    crt_flag = 1;
                }else if(neighbors_array[unrollt_id].second == 0){
                    if(check_zero == 0){
                        check_zero = 1;
                        crt_flag = 1;
                    }
                }
            }
            tmp_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(tmp_flag != 0){
                break;
            }else if(iteration == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }
        
        // if(hash_iteration == 4){
        //     for(int i = 0; i < (size32 + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < size32){
        //             data[unrollt_id] = 0;
        //         }
        //     }
        //     __syncthreads();
            
        //     for(int i = 0; i < (num_of_candidates * num_hash + size_of_block - 1) / size_of_block; i++){
        //         int unrollt_id = t_id + i * size_of_block;
        //         if(unrollt_id < num_of_candidates * num_hash){
        //             int index = abs(neighbors_array[unrollt_id / num_hash].second);
        //             if(index < total_num_of_points){
        //                set_bit(hash_(unrollt_id % num_hash,index,random_number),data); 
        //             }
        //         }
        //     }
        //     //__syncthreads();
        //     hash_iteration = 0;
        // }
       
        // hash_iteration++;
    }
    
    for (int i = 0; i < (num_of_results + size_of_block - 1) / size_of_block; i++) {
        int unrollt_id = t_id + size_of_block * i;
    
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id] = abs(neighbors_array[unrollt_id].second);
            // crt_dis[unrollt_id] = neighbors_array[unrollt_id].first;
            
        }
    }
}
