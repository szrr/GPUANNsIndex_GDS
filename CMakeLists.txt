cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)  # Specify the languages explicitly

# Enable CUDA support
find_package(CUDA REQUIRED)

# Add the directories for the headers
include_directories(
    ${PROJECT_SOURCE_DIR}
    functions
    graph/graph_index
    graph/graph_kernel_operation
    graph/subgraph_build_merge
    # hybrid
    rvq
    utils
    storage_engine/include/freestanding/include
    storage_engine/include
)

add_definitions(-DUSE_L2_DIST_)

# Add the giann_core library path
link_directories(${PROJECT_SOURCE_DIR}/storage_engine/build/lib)

# Add source files (make sure to include at least one .cu or .cpp file)
set(SRC_FILES
    query.cu
    functions/distance_kernel.cu
    # functions/Kselect.cu
    functions/selectMin1.cu
    graph/graph_index/nsw_graph_operations.cu
    graph/subgraph_build_merge/subgraph_operation.cu
    RVQ/RVQ.cu
)


# Specify the executable or library to build
add_executable(query-client ${SRC_FILES})


# Link against the libnvm library from giann_core
target_link_libraries(query-client PRIVATE 
                        ${PROJECT_SOURCE_DIR}/storage_engine/build/lib/libnvm.so
                        cublas
                        openblas)

# Set CUDA-specific compilation options
set_target_properties(query-client PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70"  # Adjust according to your GPU architecture
)

# Ensure the correct path to the cuBLAS library is linked
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(query-client PRIVATE ${CUDA_LIBRARIES})

set(SRC_FILES_G2L
    ${CMAKE_SOURCE_DIR}/graph2libnvm.cu
)

add_executable(g2l-client ${SRC_FILES_G2L})
target_link_libraries(g2l-client PRIVATE ${CMAKE_SOURCE_DIR}/storage_engine/build/lib/libnvm.so)
set_target_properties(g2l-client PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70"
)