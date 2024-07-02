
#include "gemm.h"

cudaError_t GemmGPU_V1(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta) {

    auto err = cudaGetLastError();
    return err;
}