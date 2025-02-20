// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <sys/stat.h>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define METADATA_SIZE 4096
#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)
inline void get_bin_metadata_impl(std::basic_istream<char> &reader, size_t &nrows, size_t &ncols, size_t offset = 0)
{
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char *)&nrows_32, sizeof(int));
    reader.read((char *)&ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
}

inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0)
{
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols, offset);
}

template <typename T>
inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim, size_t file_offset = 0)
{
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    // std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    data = new T[npts * dim];
    reader.read((char *)data, npts * dim * sizeof(T));
}

template <typename T>
inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0)
{
    // std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
    //    std::cout << "Opening bin file " << bin_file.c_str() << "... " << std::endl;
        reader.open(bin_file, std::ios::binary | std::ios::ate);
        reader.seekg(0);
        load_bin_impl<T>(reader, data, npts, dim, offset);
    }
    catch (std::system_error &e)
    {
        std::cout << "Cant opening bin file " << bin_file.c_str() << "... " << std::endl;
        return;
    }
    // std::cout << "done." << std::endl;
}

class PQ
{
public:
    uint8_t *data = nullptr; //compressed vector data
    float *tables = nullptr; // pq_tables = float array of size [256 * ndims]
    int64_t num_of_points = 0;
    uint64_t ndims = 0;      // ndims = true dimension of vectors
    uint64_t num_of_chunks = 0;
    uint32_t *chunk_offsets = nullptr;
    float *centroid = nullptr;
    float *tables_tr = nullptr; // same as pq_tables, but col-major

    PQ(){};

    ~PQ();

    void load_from_separate_paths(const char *pivots_filepath, const char *compressed_filepath);

};

void PQ::load_from_separate_paths(const char *pivots_filepath, const char *compressed_filepath){
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    size_t pq_file_dim, pq_file_num_centroids;

    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
    this->ndims = pq_file_dim;

    size_t npts_u64, nchunks_u64;
    load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);

    this->num_of_points = npts_u64;
    this->num_of_chunks = nchunks_u64;
    // std::cout << "Loaded PQ compressed data:" << ", number of points: " << this->num_of_points
    //               << ", chunks: " << this->num_of_chunks << std::endl;
     uint64_t nr, nc;
    size_t* file_offset_data;
    load_bin<size_t>(pq_table_bin, file_offset_data, nr, nc);
    bool use_old_filetype = false;

    if (nr != 4 && nr != 5)
    {
        std::cout << "Error reading pq_pivots file " << pq_table_bin
                      << ". Offsets dont contain correct metadata, # offsets = " << nr << ", but expecting " << 4
                      << " or " << 5;
        return;
    }

    if (nr == 4)
    {
    //    std::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
    //                   << " " << file_offset_data[3] << std::endl;
    }
    else if (nr == 5)
    {
        use_old_filetype = true;
        // std::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " " << file_offset_data[2]
        //               << " " << file_offset_data[3] << " " <<file_offset_data[4] << std::endl;
    }
    else
    {
        std::cout << "Wrong number of offsets in pq_pivots" << std::endl;
        return;
    }
    load_bin<float>(pq_table_bin, tables, nr, nc, file_offset_data[0]);
    if ((nr != NUM_PQ_CENTROIDS))
    {
        std::cout << "Error reading pq_pivots file " << pq_table_bin << ". file_num_centers  = " << nr
                      << " but expecting " << NUM_PQ_CENTROIDS << " centers";
        return;
    }

    load_bin<float>(pq_table_bin, centroid, nr, nc, file_offset_data[1]);
    if ((nr != this->ndims) || (nc != 1))
    {
        std::cerr << "Error reading centroids from pq_pivots file " << pq_table_bin << ". file_dim  = " << nr
                      << ", file_cols = " << nc << " but expecting " << this->ndims << " entries in 1 dimension.";
        return;
    }

    int chunk_offsets_index = 2;
    if (use_old_filetype)
    {
        chunk_offsets_index = 3;
    }
    load_bin<uint32_t>(pq_table_bin, chunk_offsets, nr, nc, file_offset_data[chunk_offsets_index]);
    if (nc != 1 || (nr != num_of_chunks + 1 && num_of_chunks != 0))
    {
        std::cerr << "Error loading chunk offsets file. numc: " << nc << " (should be 1). numr: " << nr
                      << " (should be " << num_of_chunks + 1 << " or 0 if we need to infer)" << std::endl;
        return;
    }
    // std::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << this->ndims
    //               << ", #chunks: " << this->num_of_chunks << std::endl;
    tables_tr = new float[256 * this->ndims];
    for (size_t i = 0; i < 256; i++)
    {
        for (size_t j = 0; j < this->ndims; j++)
        {
            tables_tr[j * 256 + i] = tables[i * this->ndims + j];
        }
    }
}