#include "gemm.h"

#include <cuda_runtime.h>

#include "common/common.h"


#ifdef DIVUP
#undef DIVUP
#define DIVUP(x, y) (x -1) / y + 1
#endif

#define THREADPERBLOCK 512


__global__ void GemmGPU_V0_Kernel(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta) {
    size_t th_idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t i = th_idx / N;
    size_t j = th_idx % N;
    float tmp_sum = 0.f;
    for (size_t k = 0; k < K; k++) {
        float a = A[i * K + k];
        float b = B[k * N + j];
        tmp_sum += a*b; 
    }
    C[th_idx] = alpha * tmp_sum + beta * C[th_idx];
    return;
}


cudaError_t GemmGPU_V0(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta)
{
    // 每个线程计算一个C的元素
    dim3 block_num = DIVUP(M*N, THREADPERBLOCK);
    GemmGPU_V0_Kernel<<<block_num, THREADPERBLOCK>>>(A, B, C, M, N, K, TransA, TransB, alpha, beta);
    auto err = cudaGetLastError();
    return err;
}