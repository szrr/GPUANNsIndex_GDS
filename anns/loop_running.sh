#!/bin/bash

export PATH=$PATH:/usr/local/cuda-10.0/bin

for i in 1 2 3 4 5 6 7 8 9 10
do
	./query_128_l2 /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /mnt/data1/szr/dataset/sift1b/bigann_query.bvecs nsw /home/ErHa/GANNS_Res/bigann_base.bvecs_64_32_1M.nsw /mnt/data1/szr/dataset/sift1b/gnd/idx_1M.ivecs 32 10 1000000
done