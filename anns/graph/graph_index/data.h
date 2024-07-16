#pragma once

#include <memory>
#include <vector>
#include <string.h>
#include <fstream>
#include <cmath>
#include "../../common.h"

using namespace std;

class Data{

public:
    float* data_;
    int num_of_points_;
    int dim_of_point_;

    /*void ReadVectorsFromFiles(string path){
        ifstream in_descriptor(path, std::ios::binary);
        
        if (!in_descriptor.is_open()) {
            exit(1);
        }

        in_descriptor.read((char*)&dim_of_point_, 4);

        in_descriptor.seekg(0, std::ios::end);
        long long file_size = in_descriptor.tellg(); 
        num_of_points_ = file_size / (dim_of_point_ + 1) / 4;

        data_ = new float[dim_of_point_ * num_of_points_];
        //memset(data_, 0, 4 * num_of_points_ * dim_of_point_);
    
        in_descriptor.seekg(0, std::ios::beg);

        for (int i = 0; i < num_of_points_; i++) {
            in_descriptor.seekg(4, std::ios::cur);
            in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_ * 4);
        }

        in_descriptor.close();
    }*/
    void ReadVectorsFromFiles(string path){
        ifstream in_descriptor(path, std::ios::binary);
        
        if (!in_descriptor.is_open()) {
            exit(1);
        }
        int dim;
        in_descriptor.read((char*)&dim, 4);
        dim_of_point_=dim;
        in_descriptor.seekg(0, std::ios::end);
        long long file_size = in_descriptor.tellg(); 
        num_of_points_ = file_size / (dim_of_point_ + 4);
        cout<<"Dim: "<<dim_of_point_<<" num_of_points: "<<num_of_points_<<endl;
        data_ = new float[dim_of_point_ * num_of_points_];
        //memset(data_, 0, 4 * num_of_points_ * dim_of_point_);
    
        in_descriptor.seekg(0, std::ios::beg);

        for (int i = 0; i < num_of_points_; i++) {
            unsigned char tmp_data[dim_of_point_];
            in_descriptor.seekg(4, std::ios::cur);
            //in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_);
            in_descriptor.read((char*)(tmp_data), dim_of_point_);
            for(int l = 0;l < dim_of_point_; l++){
                data_[i * dim_of_point_ + l] = float(tmp_data[l]);
            }
        }

        in_descriptor.close();
    }

    void ReadVectorsFromFiles(string path,int n){
        ifstream in_descriptor(path, std::ios::binary);
        
        if (!in_descriptor.is_open()) {
            exit(1);
        }

        int dim;
        in_descriptor.read((char*)&dim, 4);
        dim_of_point_=dim;
        printf("Dim:%d \n ",dim_of_point_);
        //in_descriptor.seekg(0, std::ios::end);
        //long long file_size = in_descriptor.tellg(); 
        num_of_points_ = n;
        data_ = new float[dim_of_point_ * num_of_points_];
        //memset(data_, 0, 4 * num_of_points_ * dim_of_point_);
    
        in_descriptor.seekg(0, std::ios::beg);

        for (int i = 0; i < num_of_points_; i++) {
            unsigned char tmp_data[dim_of_point_];
            in_descriptor.seekg(4, std::ios::cur);
            //in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_);
            in_descriptor.read((char*)(tmp_data), dim_of_point_);
            for(int l = 0;l < dim_of_point_; l++){
                data_[i * dim_of_point_ + l] = float(tmp_data[l]);
                //data_[i * dim_of_point_ + l]=1;
            }
        }
        in_descriptor.close();
    }

public:
    Data(string path){
        ReadVectorsFromFiles(path);
    }

    Data(string path,int n){
        ReadVectorsFromFiles(path,n);
    }
    
    float* GetFirstPositionofPoint(int point_id) {
        return data_ + point_id * dim_of_point_;
    }

    float L2Distance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return sqrt(dist);
    }
    
    float IPDistance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            dist -= a[i] * b[i];
        }
        return dist;
    }
    
    float COSDistance(float* a, float* b) {
        float dist = 0;
        float length_a = 0, length_b = 0;

        for (int i = 0; i < dim_of_point_; ++i) {
            dist += a[i] * b[i];
            length_a += a[i] * a[i];
            length_b += b[i] * b[i];
        }

        dist = dist / sqrt(length_a) / sqrt(length_b);

        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }

        return dist;
    }

    inline int GetNumPoints(){
        return num_of_points_;
    }

    int GetDimofPoints(){
        return dim_of_point_;
    }

};