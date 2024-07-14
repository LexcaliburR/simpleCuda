#pragma once

#include <cuda_runtime.h>
#include <vector>


void TransposeCPU(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis);
void TransposeCPUOPENMP(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis);
cudaError_t TransposeGPUNaive(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis);
cudaError_t TransposeGPUShared(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis);
cudaError_t TransposeGPUSharedV2(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis);
