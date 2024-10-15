#pragma once
#include "../bam_impl.h"

__global__ void writePreGraph2SSD(array_d_t<_Float32>* da, _Float32* data, uint32_t NUM_ITEMS) {
    // 计算当前线程的全局索引
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程可以处理多个批次的向量
    for (uint32_t i = idx; i < NUM_ITEMS; i += blockDim.x * gridDim.x) {
        // 获取当前线程负责的 128 维向量的起始地址
        _Float32* item = &data[i * 128];

        // 分配一个新的数组用于存储 128 维向量 + 128 个随机数
        _Float32 expanded_item[256];  // 128 维向量 + 128 个随机数

        // 将 128 维向量复制到 expanded_item 的前 128 个位置
        for (int j = 0; j < 128; ++j) {
            expanded_item[j] = item[j];
        }

        // 生成 128 个随机数并放入 expanded_item 的后半部分
        for (int j = 128; j < 256; ++j) {
            expanded_item[j] = -1;// 生成 0 到 1 之间的随机数，最终应该在这里调用生成R正则图的方法
        }

        // 将这个扩展的 256 长度的数组写入磁盘
        // printf("call item_write...\n");
        da->item_write(i, expanded_item, 256);
    }
}


__global__ void readTest(array_d_t<_Float32>* da, uint32_t NUM_ITEMS) {
    // 计算当前线程的全局索引
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程可以处理多个批次的向量
    for (uint32_t i = idx; i < NUM_ITEMS; i += blockDim.x * gridDim.x) {
        printf("%f\n",da->vec_access(i, 0, 256));
    }
}