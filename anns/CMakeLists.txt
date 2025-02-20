cmake_minimum_required(VERSION 3.23)

project(ANNSBaM LANGUAGES CXX CUDA C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package (CUDA 8.0)

add_definitions(-D__ATOMIC_THREAD=10)

set (nvidia_archs "70;80;90" CACHE STRING "NVIDIA compute architectures")
set (no_smartio_samples true CACHE BOOL "Do not use SISCI for examples")
set (no_smartio_benchmarks true CACHE BOOL "Do not use SISCI for benchmarks")
set (no_smartio true CACHE BOOL "Do not use SISCI for library or examples")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (no_cc false CACHE BOOL "Do not assume cache-coherent architecture")
set (no_fio true CACHE BOOL "Do not build FIO plugin")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")




# Ensure MKLROOT is set
if(NOT DEFINED ENV{MKLROOT})
  message(FATAL_ERROR "MKLROOT environment variable is not set. Please set it to your MKL installation path.")
endif()

# Set MKL include and library directories
set(MKL_INCLUDE_DIR "$ENV{MKLROOT}/include")
set(MKL_LIBRARY_DIR "$ENV{MKLROOT}/lib/intel64")

# Include MKL headers
include_directories(${MKL_INCLUDE_DIR})

# Link MKL libraries
link_directories(${MKL_LIBRARY_DIR})

set(BAM_DIR "${CMAKE_SOURCE_DIR}/bam")
set(BAM_LIB_DIR "${BAM_DIR}/build/lib")
set(BAM_INCLUDE_DIR "${BAM_DIR}/include")


set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11 -fopenmp -pthread" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++11" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -O3;-std=c++11;${GPU_ARCHS}")
set (CUDA_NVCC_FLAGS_RELEASE "${CMAKE_NVCC_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)


add_subdirectory(${BAM_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)

link_directories(${BAM_LIB_DIR})

# add_subdirectory(functions)

message(${CMAKE_SOURCE_DIR})

include_directories(${BAM_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/hybrid)
include_directories(${CMAKE_SOURCE_DIR}/graph/graph_index)
include_directories(${CMAKE_SOURCE_DIR}/RVQ)
include_directories(${CMAKE_SOURCE_DIR}/functions)
include_directories(${CMAKE_SOURCE_DIR}/graph/graph_kernel_operation)
include_directories(${CMAKE_SOURCE_DIR}/bam/include/freestanding/include)

set(BAM_DIR "${CMAKE_SOURCE_DIR}/bam")
set(BAM_LIB_DIR "${BAM_DIR}/build/lib")
set(BAM_INCLUDE_DIR "${BAM_DIR}/include")
set(BAM_SRC_DIR "${BAM_DIR}/src")

include_directories(${BAM_INCLUDE_DIR})
include_directories(${BAM_INCLUDE_DIR}/freestanding/include)
include_directories(${BAM_SRC_DIR}/linux)
include_directories(${BAM_SRC_DIR})

link_directories(${BAM_LIB_DIR})

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -O3;-std=c++11;${GPU_ARCHS}")

file(GLOB BAM_SRC_FILES
    "${BAM_SRC_DIR}/*.cpp"
    "${BAM_SRC_DIR}/dis/*.cu"
    "${BAM_SRC_DIR}/linux/*.cpp"
)
# Search
set(SOURCE_FILES 
    ${BAM_SRC_FILES}
    # ${CMAKE_SOURCE_DIR}/hybrid/hybrid.cpp
    ${CMAKE_SOURCE_DIR}/query.cu
    ${CMAKE_SOURCE_DIR}/RVQ/RVQ.cu
    #${CMAKE_SOURCE_DIR}/graph/graph_index/nsw_graph_operations.cu
    ${CMAKE_SOURCE_DIR}/functions/distance_kernel.cu
    ${CMAKE_SOURCE_DIR}/functions/selectMin1.cu
)

add_executable(query ${SOURCE_FILES})

target_link_libraries(query libnvm ${CUDA_LIBRARIES} -lcublas -L/opt/intel/oneapi/mkl/2024.2/lib -lmkl_rt -fopenmp)

target_compile_definitions(query PRIVATE USE_L2_DIST_ _CUDA __CUDA__)

set_target_properties(query PROPERTIES CUDA_ARCHITECTURES "70;80;90")

#Build
# set(SOURCE_FILES 
#     ${BAM_SRC_FILES}
#     ${CMAKE_SOURCE_DIR}/build.cu
#     # ${CMAKE_SOURCE_DIR}/graph/graph_index/nsw_graph_operations.cu
# )

# add_executable(build ${SOURCE_FILES})

# target_link_libraries(build libnvm ${CUDA_LIBRARIES} -lcublas -L/opt/intel/oneapi/mkl/2024.2/lib -lmkl_rt -fopenmp)

# target_compile_definitions(build PRIVATE USE_L2_DIST_ _CUDA __CUDA__)

# set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "70;80;90")