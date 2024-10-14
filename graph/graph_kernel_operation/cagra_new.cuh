#pragma once

#include<math.h>
#include "structure_on_device.cuh"
#include "warpselect/WarpSelect.cuh"
#define size32 512
#define shift 9
#define num_hash 2

// __device__
// int pure_hash(int h,uint32_t x,uint32_t* random_number){ 
//     x ^= x >> 16;
//     x *= random_number[h << 1];
//     x ^= x >> 13;
//     x *= random_number[(h << 1) + 1];
//     x ^= x >> 16;
//     return x;
// }

__device__
int hash_(int h,uint32_t x,uint32_t* random_number){ 
    x ^= x >> 16;
    x *= random_number[h];
    x ^= x >> 13;
    x *= random_number[(h) + 1];
    x ^= x >> 16;
    //return x & 31;
    return x % ((size32 << 5));
}

__device__
void set_bit(int x,uint32_t* data){
    unsigned int mask = 1U << (x / size32);
    atomicOr(&data[x % size32], mask);
}

__device__
bool test_bit(int offset,int x,uint32_t* data){
    //return ((data[offset] >> x) & 1);
    return ((data[x % size32] >> (x / size32)) & 1);
}

// __device__
// int get_offset(uint32_t x,uint32_t* random_number){
//     return (pure_hash(9,x,random_number) & (size32 - 1));
// }

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
template <typename IdType, typename FloatType, int WARP_SIZE, int NumWarpQ, int NumThreadQ>
__global__ void cagra(float* d_data, float* d_query, int* d_result, int* d_graph, int total_num_of_points, int offset_shift, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int search_width,
                    int* d_enter_cluster, int** d_rvq_indices, int* d_rvq_indices_size){
    #define DIM 128
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;
    int size_of_block = blockDim.x;
    int lane_id = threadIdx.x % size_of_warp;
    int warp_id = threadIdx.x / size_of_warp;
    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
    uint32_t* data = (uint32_t*)(shared_memory_space_s + (search_width << offset_shift));
    __shared__ int tmp_flag_all_blocks;
    
    int* crt_result = d_result + b_id * num_of_results;
    int degree_of_point = (1 << offset_shift);
    int neighbors_array_len = search_width << offset_shift;

    // __shared__ uint32_t data[size32];
    // uint32_t random_number[10 * 2] = {
	// 	0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
	// 	0x1bead407U,0x28a30449U,0xbfc4d99fU,
	// 	0x715030e2U,0xffcfb45bU,0x6e4ce166U,
	// 	0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
	// 	0x0288592dU,0x362c37bcU,0x9d4824f0U,
	// 	0xfdbdd68bU,0x63258c85U,0x6726905cU,
	// 	0x609500f9U,0x4de48422U
    // };
    uint32_t random_number[4] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U
    };

    WarpSelect<float, int, false, Comparator<float>, NumWarpQ, NumThreadQ, WARP_SIZE>heap(MAX, total_num_of_points, num_of_candidates);

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
    int cluster_id = d_enter_cluster[b_id];
    int enter_points_num = min(d_rvq_indices_size[cluster_id], neighbors_array_len);
    int* enter_points_pos = d_rvq_indices[cluster_id];
    int iteration;

    for(int i = warp_id; i < enter_points_num; i += 2){
        int target_point_id = enter_points_pos[i];
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = d_data[target_point_id * DIM + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = d_data[target_point_id * DIM + lane_id + 32];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = d_data[target_point_id * DIM + lane_id + 64];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = d_data[target_point_id * DIM + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = d_data[target_point_id * DIM + lane_id + 128];
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = d_data[target_point_id * DIM + lane_id + 160];
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = d_data[target_point_id * DIM + lane_id + 192];
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = d_data[target_point_id * DIM + lane_id + 224];
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = d_data[target_point_id * DIM + lane_id + 256];
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = d_data[target_point_id * DIM + lane_id + 288];
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = d_data[target_point_id * DIM + lane_id + 320];
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = d_data[target_point_id * DIM + lane_id + 352];
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = d_data[target_point_id * DIM + lane_id + 384];
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = d_data[target_point_id * DIM + lane_id + 416];
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = d_data[target_point_id * DIM + lane_id + 448];
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = d_data[target_point_id * DIM + lane_id + 480];
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = d_data[target_point_id * DIM + lane_id + 512];
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = d_data[target_point_id * DIM + lane_id + 544];
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = d_data[target_point_id * DIM + lane_id + 576];
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = d_data[target_point_id * DIM + lane_id + 608];
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = d_data[target_point_id * DIM + lane_id + 640];
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = d_data[target_point_id * DIM + lane_id + 672];
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = d_data[target_point_id * DIM + lane_id + 704];
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = d_data[target_point_id * DIM + lane_id + 736];
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = d_data[target_point_id * DIM + lane_id + 768];
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = d_data[target_point_id * DIM + lane_id + 800];
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = d_data[target_point_id * DIM + lane_id + 832];
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = d_data[target_point_id * DIM + lane_id + 864];
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = d_data[target_point_id * DIM + lane_id + 896];
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = d_data[target_point_id * DIM + lane_id + 928];
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
        if(lane_id == 0){
            neighbors_array[i].first = dist;
            neighbors_array[i].second = target_point_id;
        }
    }
    // __threadfence_block();
    __syncthreads();
    if(warp_id == 1){
        for(int i = 0; i < (enter_points_num + size_of_warp - 1) / size_of_warp; i++){
            int unrollt_id = lane_id + i * size_of_warp;
            if(unrollt_id < enter_points_num){
                add(neighbors_array[unrollt_id].second, random_number, data);
            }
        }
    }
    if(warp_id == 0){
        for(int l = 0; l < (enter_points_num + size_of_warp - 1) / size_of_warp; l++){
            int unrollt_id = lane_id + l * size_of_warp;
            if(unrollt_id < enter_points_num){
                heap.addThreadQ(neighbors_array[unrollt_id].first, neighbors_array[unrollt_id].second);
            }
        }
        heap.reduce();
    }
    __syncthreads();
    int flag_all_blocks = 1;
    int tmp_flag = (1 << min(min(num_of_explored_points, size_of_warp), enter_points_num)) - 1;
    
    int tmp_search_width;
    int first_position_of_flag;
    int check_zero = 0;
    int hash_iteration = 0;
    iteration = 0;
    int d_iter = 0;
    int max_iter = num_of_explored_points / search_width;
    while (flag_all_blocks && d_iter < max_iter){
        d_iter++;
        tmp_search_width = 0;
        for(int i = 0; i < (neighbors_array_len + size_of_block - 1) / size_of_block; i++){
            int unrollt_id = t_id + size_of_block * i;
            if(unrollt_id < neighbors_array_len){
                neighbors_array[unrollt_id].second = -1;
                neighbors_array[unrollt_id].first = MAX;
            }
        }
        // __threadfence_block();
        __syncthreads();
        if(warp_id == 0){
            while(tmp_search_width < search_width && tmp_flag != 0){
                int first_position_of_tmp_flag = __ffs(tmp_flag) - 1;
                first_position_of_flag = __shfl_sync(0xFFFFFFFF, heap.warpV[iteration], first_position_of_tmp_flag);
                if(lane_id == first_position_of_tmp_flag){
                    heap.warpV[iteration] = -heap.warpV[iteration];
                }
                tmp_flag &= ~(1 << first_position_of_tmp_flag);
                //读取邻居
                // if(first_position_of_flag < 0 || first_position_of_flag >= total_num_of_points){
                //     printf("first_position_of_flag: %d \n", first_position_of_flag);
                // }
                auto offset = first_position_of_flag << offset_shift;
                int* neighbors_of_first_point = d_graph + offset;
                for(int i = 0; i < (degree_of_point + size_of_warp - 1) / size_of_warp; i++){
                    int unrollt_id = lane_id + i * size_of_warp;
                    if(unrollt_id < degree_of_point){
                        int loc = tmp_search_width * degree_of_point + unrollt_id;
                        neighbors_array[loc].second = neighbors_of_first_point[unrollt_id];
                        if(neighbors_array[loc].second == total_num_of_points || test(neighbors_array[loc].second, random_number, data)){  //|| test(neighbors_array[loc].second, random_number, data)
                            neighbors_array[loc].first = MAX;
                        }else{
                            neighbors_array[loc].first = 0;
                            add(neighbors_array[loc].second, random_number, data);
                        }
                    }
                }
                tmp_search_width++;
                if(tmp_search_width == search_width) break;

                while(tmp_flag == 0 && iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp){
                    iteration++;
                    int unrollt_id = lane_id + size_of_warp * iteration;
                    int crt_flag = 0;
                    if(unrollt_id < num_of_explored_points){
                        if(heap.warpV[iteration] > 0 && heap.warpV[iteration] != total_num_of_points){
                            crt_flag = 1;
                        }else if(heap.warpV[iteration] == 0){
                            if(check_zero == 0){
                                check_zero = 1;
                                crt_flag = 1;
                            }
                        }
                    }
                    tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
                }
            }
            // __syncthreads();
            // for(int i = 0; i < NumWarpQ / kWarpSize; i++){
            //     for(int l = 0; l < tmp_search_width * degree_of_point; l++){
            //         if(abs(heap.warpV[i]) == neighbors_array[l].second){
            //             neighbors_array[l].first = MAX;
            //         }
            //     }
            // }
        }
        __syncthreads();
        if(neighbors_array[0].second == -1) break;
        //计算距离
        for(int i = warp_id; i < neighbors_array_len; i+=2){
            int target_point_id = neighbors_array[i].second;
            // if(target_point_id < -1 || target_point_id > total_num_of_points){
            //     printf("target_point_id:%d \n", target_point_id);
            // }
            if(target_point_id == -1) break;
            if(target_point_id == total_num_of_points || neighbors_array[i].first == MAX) continue;
#if DIM > 0
float p1 = 0;
if (lane_id < DIM) {
    p1 = d_data[target_point_id * DIM + lane_id];
}
#endif
#if DIM > 32
float p2 = 0;
if (lane_id + 32 < DIM) {
    p2 = d_data[target_point_id * DIM + lane_id + 32];
}
#endif
#if DIM > 64
float p3 = 0;
if (lane_id + 64 < DIM) {
    p3 = d_data[target_point_id * DIM + lane_id + 64];
}
#endif
#if DIM > 96
float p4 = 0;
if (lane_id + 96 < DIM) {
    p4 = d_data[target_point_id * DIM + lane_id + 96];
}
#endif
#if DIM > 128
float p5 = 0;
if (lane_id + 128 < DIM) {
    p5 = d_data[target_point_id * DIM + lane_id + 128];
}
#endif
#if DIM > 160
float p6 = 0;
if (lane_id + 160 < DIM) {
    p6 = d_data[target_point_id * DIM + lane_id + 160];
}
#endif
#if DIM > 192
float p7 = 0;
if (lane_id + 192 < DIM) {
    p7 = d_data[target_point_id * DIM + lane_id + 192];
}
#endif
#if DIM > 224
float p8 = 0;
if (lane_id + 224 < DIM) {
    p8 = d_data[target_point_id * DIM + lane_id + 224];
}
#endif
#if DIM > 256
float p9 = 0;
if (lane_id + 256 < DIM) {
    p9 = d_data[target_point_id * DIM + lane_id + 256];
}
#endif
#if DIM > 288
float p10 = 0;
if (lane_id + 288 < DIM) {
    p10 = d_data[target_point_id * DIM + lane_id + 288];
}
#endif
#if DIM > 320
float p11 = 0;
if (lane_id + 320 < DIM) {
    p11 = d_data[target_point_id * DIM + lane_id + 320];
}
#endif
#if DIM > 352
float p12 = 0;
if (lane_id + 352 < DIM) {
    p12 = d_data[target_point_id * DIM + lane_id + 352];
}
#endif
#if DIM > 384
float p13 = 0;
if (lane_id + 384 < DIM) {
    p13 = d_data[target_point_id * DIM + lane_id + 384];
}
#endif
#if DIM > 416
float p14 = 0;
if (lane_id + 416 < DIM) {
    p14 = d_data[target_point_id * DIM + lane_id + 416];
}
#endif
#if DIM > 448
float p15 = 0;
if (lane_id + 448 < DIM) {
    p15 = d_data[target_point_id * DIM + lane_id + 448];
}
#endif
#if DIM > 480
float p16 = 0;
if (lane_id + 480 < DIM) {
    p16 = d_data[target_point_id * DIM + lane_id + 480];
}
#endif
#if DIM > 512
float p17 = 0;
if (lane_id + 512 < DIM) {
    p17 = d_data[target_point_id * DIM + lane_id + 512];
}
#endif
#if DIM > 544
float p18 = 0;
if (lane_id + 544 < DIM) {
    p18 = d_data[target_point_id * DIM + lane_id + 544];
}
#endif
#if DIM > 576
float p19 = 0;
if (lane_id + 576 < DIM) {
    p19 = d_data[target_point_id * DIM + lane_id + 576];
}
#endif
#if DIM > 608
float p20 = 0;
if (lane_id + 608 < DIM) {
    p20 = d_data[target_point_id * DIM + lane_id + 608];
}
#endif
#if DIM > 640
float p21 = 0;
if (lane_id + 640 < DIM) {
    p21 = d_data[target_point_id * DIM + lane_id + 640];
}
#endif
#if DIM > 672
float p22 = 0;
if (lane_id + 672 < DIM) {
    p22 = d_data[target_point_id * DIM + lane_id + 672];
}
#endif
#if DIM > 704
float p23 = 0;
if (lane_id + 704 < DIM) {
    p23 = d_data[target_point_id * DIM + lane_id + 704];
}
#endif
#if DIM > 736
float p24 = 0;
if (lane_id + 736 < DIM) {
    p24 = d_data[target_point_id * DIM + lane_id + 736];
}
#endif
#if DIM > 768
float p25 = 0;
if (lane_id + 768 < DIM) {
    p25 = d_data[target_point_id * DIM + lane_id + 768];
}
#endif
#if DIM > 800
float p26 = 0;
if (lane_id + 800 < DIM) {
    p26 = d_data[target_point_id * DIM + lane_id + 800];
}
#endif
#if DIM > 832
float p27 = 0;
if (lane_id + 832 < DIM) {
    p27 = d_data[target_point_id * DIM + lane_id + 832];
}
#endif
#if DIM > 864
float p28 = 0;
if (lane_id + 864 < DIM) {
    p28 = d_data[target_point_id * DIM + lane_id + 864];
}
#endif
#if DIM > 896
float p29 = 0;
if (lane_id + 896 < DIM) {
    p29 = d_data[target_point_id * DIM + lane_id + 896];
}
#endif
#if DIM > 928
float p30 = 0;
if (lane_id + 224 < DIM) {
    p30 = d_data[target_point_id * DIM + lane_id + 928];
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
            if(lane_id == 0){
                neighbors_array[i].first = dist;
            }
        }
        // __threadfence_block();
        __syncthreads();
        if(warp_id == 0){
            //将候选列表与邻居列表merge之后排序
            for(int i = 0; i < (tmp_search_width * degree_of_point + size_of_warp - 1) / size_of_warp; i++){
                int unrollt_id = lane_id + i * size_of_warp;
                if(unrollt_id < tmp_search_width * degree_of_point){
                    heap.addThreadQ(neighbors_array[unrollt_id].first, neighbors_array[unrollt_id].second);
                }
            }
            heap.checkThreadQ();
            // //判断循环条件

            for (iteration = 0; iteration < (num_of_explored_points + size_of_warp - 1) / size_of_warp; iteration++) {
                int unrollt_id = lane_id + size_of_warp * iteration;
                int crt_flag = 0;

                if(unrollt_id < num_of_explored_points){
                    if(heap.warpV[iteration] > 0 && heap.warpV[iteration] != total_num_of_points){
                        crt_flag = 1;
                    }else if(heap.warpV[iteration] == 0){
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
            if(t_id == 0){
                tmp_flag_all_blocks = flag_all_blocks;
            }
        }
        __syncthreads();
        if(warp_id == 1){
            flag_all_blocks = tmp_flag_all_blocks;
        }
        if(flag_all_blocks == 0) break;
        
        if(hash_iteration == 4){
            for(int i = 0; i < (size32 + size_of_block - 1) / size_of_block; i++){
                int unrollt_id = t_id + i * size_of_block;
                if(unrollt_id < size32){
                    data[unrollt_id] = 0;
                }
            }
            __syncthreads();
            if(warp_id == 0){
                for(int i = 0; i < (num_of_candidates + size_of_warp - 1) / size_of_warp; i++){
                    int unrollt_id = lane_id + i * size_of_warp;
                    if(unrollt_id < num_of_candidates){
                        int index_id = abs(heap.warpV[i]);
                        if(index_id < total_num_of_points){
                            add(index_id, random_number, data);
                        }
                    }
                }
            }
            hash_iteration = 0;
        }
       
         hash_iteration++;
    }
    
    // __syncthreads();
    if(warp_id == 0){
        for (int i = 0; i < (num_of_results + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = lane_id + size_of_warp * i;
        
            if (unrollt_id < num_of_results) {
                crt_result[unrollt_id] = abs(heap.warpV[i]);
            }
        }
    }
}