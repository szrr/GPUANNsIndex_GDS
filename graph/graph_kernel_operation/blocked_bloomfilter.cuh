#pragma once
#include "../../common.h"
#define GPU_CACHE_LINE_SIZE64 1
#define GPU_CACHE_LINE_SHIFT 0
#define BLOOMFILTER_DATA_T uint32_t
#define BLOOMFILTER_SIZE64MULT 2
#define BLOOMFILTER_SIZE_SHIFT 5

template<const int size64,const int shift,const int num_hash>
struct BlockedBloomFilter{
    BLOOMFILTER_DATA_T* data;
    //const static int num_hash = 7;
	
	const uint32_t random_number[10 * 2] = {
		0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
		0x1bead407U,0x28a30449U,0xbfc4d99fU,
		0x715030e2U,0xffcfb45bU,0x6e4ce166U,
		0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
		0x0288592dU,0x362c37bcU,0x9d4824f0U,
		0xfdbdd68bU,0x63258c85U,0x6726905cU,
		0x609500f9U,0x4de48422U
        };
    
	__device__
	BlockedBloomFilter(){
		data = new BLOOMFILTER_DATA_T[size64 * BLOOMFILTER_SIZE64MULT];
        for(int i = 0;i < size64;++i)
            data[i] = 0;
    }


	__device__
	void reset(){
		for(int i = 0;i < size64 * BLOOMFILTER_SIZE64MULT;++i)
            data[i] = 0;
	}

	__device__
    int pure_hash(int h,idx_t x){ 
		x ^= x >> 17;
		x *= random_number[h << 1];
		x ^= x >> 17;
		x *= random_number[(h << 1) + 1];
		x ^= x >> 17;
		return x;
	}

	__device__
    int hash(int h,idx_t x){ 
		x ^= x >> 17;
		x *= random_number[h << 1];
		x ^= x >> 17;
		x *= random_number[(h << 1) + 1];
		x ^= x >> 17;
    	return x & ((GPU_CACHE_LINE_SIZE64 << BLOOMFILTER_SIZE_SHIFT) - 1);
		        //return (x ^ (x >> 32) * random_number[h << 1] ^ random_number[(h << 1) + 1]) & ((size64 << 6) - 1);
    }

	__device__
    void set_bit(int offset,int x){
        data[offset + (x & (GPU_CACHE_LINE_SIZE64 - 1))] |= (1U << (x >> GPU_CACHE_LINE_SHIFT));
    }
	
	__device__
    bool test_bit(int offset,int x){
        return ((data[offset + (x & (GPU_CACHE_LINE_SIZE64 - 1))] >> (x >> GPU_CACHE_LINE_SHIFT)) & 1);
    }

	__device__
	int get_offset(idx_t x){
		return (pure_hash(9,x) & ((size64 >> GPU_CACHE_LINE_SHIFT) - 1)) * GPU_CACHE_LINE_SIZE64;
	}

	__device__
    void add(idx_t x){
		int offset = get_offset(x);
        for(int i = 0;i < num_hash;++i)
            set_bit(offset,hash(i,x));
    }
	
	__device__
    int test(idx_t x){
		int offset = get_offset(x);
        bool ok = true;
        for(int i = 0;i < num_hash;++i)
            ok &= test_bit(offset,hash(i,x));
		if(ok) return 1;
        return 0;
    }
    
};
