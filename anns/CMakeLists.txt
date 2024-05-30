cmake_minimum_required(VERSION 3.25)

project(ANNSGDS CXX CUDA)

find_package(CUDA 12.0 REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(functions)

