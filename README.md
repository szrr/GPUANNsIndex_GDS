# GPUANNsIndex_GDS

## Functions using cuda kernel  
### done
Distance calculation: `anns/functions/distance_kernel.cu`  
Select min 1 : `anns/functions/selectMin1.cu`  
  
### under going
Select min k: `anns/functions/Kselect.cu`  

## Three-layer index
### done
Two layer RVQ index: `anns/RVQ/RVQ.cpp`  
test RVQ: `cd anns` `nvcc -o rvq RVQ.cpp ../functions/distance_kernel.cu ../functions/selectMin1.cu -lcublas -lmkl_rt`  
  
### Hybrid search
Search: `nvcc -o query query.cu ./RVQ/RVQ.cpp ./hybrid/hybrid.cpp ./graph/graph_index/nsw_graph_operations.cu ./functions/distance_kernel.cu ./functions/selectMin1.cu -lcublas -lmkl_rt -DUSE_L2_DIST_`  
test:`./query [base_path] [query_path] [graph_path] [groundtruth_path] [e] [k] [points_num]`  
For instance:`./query /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /mnt/data1/szr/dataset/sift1b/bigann_query.bvecs /home/ErHa/GANNS_Res/bigann_base.bvecs_16_8_1M.nsw /mnt/data1/szr/dataset/sift1b/gnd/idx_1M.ivecs 16 10 1000000`
## BaM data transfer
