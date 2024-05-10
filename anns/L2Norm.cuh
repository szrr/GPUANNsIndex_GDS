#pragma once


void runL2Norm(float* vectors, size_t numVec, int dims, float* res, bool NormSquared, int deviceNo, cudaStream_t stream);