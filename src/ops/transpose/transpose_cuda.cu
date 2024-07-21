#include "transpose.h"

#include <iostream>
#include "common/cuda_macros.h"

#define ThreadPerBlock 256

template <class T>
__global__ void TransposeGPUNaiveImpl(const T* f_input, T* f_output, const size_t f_num, const int* f_transStride,
                                      const int f_usedDims)
{
    size_t distIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t srcIdx = 0;
    size_t calIdx = distIdx;
    for (int i = 0; i < f_usedDims; i++) {
        size_t coord = calIdx / f_transStride[i];
        size_t left = calIdx % f_transStride[i];
        srcIdx += coord * f_transStride[i];
        calIdx = left;
    }
    f_output[distIdx] = f_input[srcIdx];
}

cudaError_t TransposeGPUNaive(const float* A, float* B, const std::vector<int>& src_shape, const std::vector<int>& axis)
{
    int eleNum = 1;

    for (auto dim : src_shape) {
        eleNum *= dim;
    }
    int* transStride = new int[src_shape.size()];
    int* inputStride = new int[src_shape.size()];
    int* transStrideDev = nullptr;
    inputStride[src_shape.size() - 1] = 1;
    for (int i = src_shape.size() - 2; i > -1; --i) {
        inputStride[i] *= inputStride[i + 1];
    }
    for (int i = 0; i < src_shape.size(); i++) {
        transStride[i] = inputStride[axis[i]];
    }
    CUDA_CHECK(cudaMalloc(&transStrideDev, sizeof(int) * src_shape.size()));
    CUDA_CHECK(cudaMemcpy(transStrideDev, transStride, sizeof(int) * src_shape.size(), cudaMemcpyHostToDevice));
    dim3 blocks(DIV(eleNum, ThreadPerBlock));
    TransposeGPUNaiveImpl<<<blocks, ThreadPerBlock>>>(A, B, eleNum, transStrideDev, src_shape.size());
    delete transStride;
    return cudaError_t();
}
cudaError_t TransposeGPUShared(const float* A, float* B, const std::vector<int>& src_shape,
                               const std::vector<int>& axis)
{
    return cudaError_t();
}
cudaError_t TransposeGPUSharedV2(const float* A, float* B, const std::vector<int>& src_shape,
                                 const std::vector<int>& axis)
{
    return cudaError_t();
}