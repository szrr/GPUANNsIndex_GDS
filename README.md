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
test:`./query [base_path] [query_path] [graph_path] [groundtruth_path] [e] [search_width] [k] [points_num] [degree_of_graph]`  
For instance:`./query ../SIFT1M ../bigann_query.bvecs ../sift1m_degree8_index.bin ../idx_1M.ivecs 256 2 10 1000000 8`   
### Hybrid build
Build:`nvcc -o build build.cu ./graph/graph_index/nsw_graph_operations.cu ./graph/subgraph_build_merge/subgraph_operation.cu -lcublas -lmkl_rt -DUSE_L2_DIST_`  
test:` `    
For instance:`./build /home/ErHa/GANNS_Res/subdata/ /home/ErHa/GANNS_Res/subgraph/ /home/ErHa/GANNS_Res/finalgraph/ 32 16 10000000`
`./build /mnt/data1/szr/dataset/sift1b/bigann_base.bvecs /home/ErHa/GANNS_Res/subgraph/ /home/ErHa/GANNS_Res/finalgraph/ 64 32 10000000`
## BaM data transfer
