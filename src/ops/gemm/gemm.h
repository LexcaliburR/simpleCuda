#pragma once

#include <cuda_runtime.h>


void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta);
cudaError_t GemmGPU_V0(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta);
cudaError_t GemmGPU_V1(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta);