# GPUANNsIndex_GDS

## update functions using cuda kernel  
### done  
Distance calculation: `anns/functions/distance_kernel.cu`  
Select min 1 : `anns/functions/selectMin1.cu`  
  
### under going
Select min k: `anns/functions/Kselect.cu`  

## update index
### done  
Two layer RVQ index: `anns/RVQ.cpp`  
test RVQ: `nvcc -o rvq RVQ.cpp ./functions/distance_kernel.cu ./functions/selectMin1.cu -lcublas -lmkl_rt`  