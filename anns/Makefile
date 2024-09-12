CXX=g++
NVCC=nvcc
DISTTYPE=USE_L2_DIST_
FLAG_DEBUG=-O3

all : build query

build : build.cu data.h graph_index/navigable_small_world.h graph_index/nsw_graph_operations.h
	$(NVCC) -ccbin g++ -I../../Common  -rdc=true -m64 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_75,code=compute_75 \
	-std=c++11 build.cu $(FLAG_DEBUG) -o build -Xptxas -v \
	-D$(DISTTYPE)

#query : query.cu graph_index/data.h graph_index/navigable_small_world.h graph_index/nsw_graph_operations.h hybrid/hybrid.h
	# $(NVCC) -ccbin g++ -I../../Common  -rdc=true -m64 \
	# -gencode=arch=compute_86,code=compute_86 \
	# -std=c++11 query.cu $(FLAG_DEBUG) -o query -Xptxas -v \
	# -D$(DISTTYPE)
query :	
		nvcc -o query query.cu -D$(DISTTYPE) -lcublas -lmkl_rt