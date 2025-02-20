# Install script for directory: /home/ErHa/GPUANNsIndex_GDS11.28/anns/bam

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/bafs_ptr.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/buffer.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/ctrl.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/event.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/host_util.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_admin.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_aq.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_cmd.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_ctrl.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_dma.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_error.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_io.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_parallel_queue.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_queue.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_rpc.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_types.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/nvm_util.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/page_cache.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/queue.h"
    "/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/include/util.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/lib/libnvm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so"
         OLD_RPATH "/opt/intel/oneapi/mkl/latest/lib/intel64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/CMakeFiles/libnvm.dir/install-cxx-module-bmi-noconfig.cmake" OPTIONAL)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/iodepth-block/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/block/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/readwrite/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/bfs/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/cc/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/pagerank/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/sssp/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/sssp_float/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/pattern/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/cache/cmake_install.cmake")
  include("/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/vectoradd/cmake_install.cmake")

endif()

