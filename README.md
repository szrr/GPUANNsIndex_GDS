# GPUANNsIndex_GDS

## Functions using cuda kernel  
### done
Distance calculation: `anns/functions/distance_kernel.cu`  
Select min 1 : `anns/functions/selectMin1.cu`  
  
### under going
Select min k: `anns/functions/Kselect.cu`  

## Three-layer index
### done
Two layer RVQ index: `anns/RVQ.cpp`  
test RVQ: `cd anns` `nvcc -o rvq RVQ.cpp ./functions/distance_kernel.cu ./functions/selectMin1.cu -lcublas -lmkl_rt`  
  
### under going
Fuse three-layer index  

## BaM data transfer