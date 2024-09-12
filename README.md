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
test RVQ: `cd anns/RVQ` `nvcc -o rvq RVQ.cu ../functions/distance_kernel.cu ../functions/selectMin1.cu -lcublas -lmkl_rt`  
  
### Hybrid search
Search: `nvcc -o query query.cu ./RVQ/RVQ.cu ./hybrid/hybrid.cpp ./graph/graph_index/nsw_graph_operations.cu ./functions/distance_kernel.cu ./functions/selectMin1.cu -lcublas -lmkl_rt -DUSE_L2_DIST_`  
test:`./query [base_path] [query_path] [graph_path] [groundtruth_path] [e] [search_width] [k] [points_num]`  
For instance:`./query /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /mnt/data1/szr/dataset/sift1b/bigann_query.bvecs /home/ErHa/GANNS_Res/graph/bigann_base.bvecs_32_16_10M.nsw /mnt/data1/szr/dataset/sift1b/gnd/idx_10M.ivecs 16 4 10 10000000`
`./query /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /mnt/data1/szr/dataset/sift1b/bigann_query.bvecs /home/ErHa/GANNS_Res/bigann_base.bvecs_64_32_10M.nsw /mnt/data1/szr/dataset/sift1b/gnd/idx_10M.ivecs 64 2 10 10000000`   
### Hybrid build
Build:`nvcc -o build build.cu ./graph/graph_index/nsw_graph_operations.cu ./graph/subgraph_build_merge/subgraph_operation.cu -lcublas -lmkl_rt -DUSE_L2_DIST_`  
test:` `    
For instance:`./build /home/ErHa/GANNS_Res/subdata/ /home/ErHa/GANNS_Res/subgraph/ /home/ErHa/GANNS_Res/finalgraph/ 32 16 10000000`
`./build /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /home/ErHa/GANNS_Res/subgraph/ /home/ErHa/GANNS_Res/finalgraph/ 64 32 10000000`
## BaM data transfer
