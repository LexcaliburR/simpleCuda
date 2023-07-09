#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <iostream>
#include "common/check_utils.h"

__global__ void ArrayAddV2(float* array1, float* array2, float* result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = array1[idx] + array2[idx];
    printf("%f", result[idx]);
    return;
}

int main()
{
    std::cout << "1111111111111" << std::endl;
    size_t size = 10000;
    int loop_times = 100;
    float* array1_h = new float[size];
    float* array2_h = new float[size];
    float* result_h = new float[size];
    for(int i = 0; i < size; ++i) {
        array1_h[i] = 1.1f;
        array2_h[i] = 2.2f;
        result_h[i] = 0.f;
    }

    float* array1_d = nullptr;
    float* array2_d = nullptr;
    float* result_d = nullptr;
    //    CUDA_CHECK(cudaMalloc((void**)&array1_d, size * sizeof(float)));
    //    CUDA_CHECK();
    cudaMalloc((void**)&array1_d, size * sizeof(float));
//    CUDA_CHECK(cudaMalloc((void**)&array2_d, size * sizeof(float)));
//    CUDA_CHECK(cudaMalloc((void**)&result_d, size * sizeof(float)));
//    CUDA_CHECK(cudaMemcpy(array1_d, array1_h, size * sizeof(float), cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemcpy(array2_d, array2_h, size * sizeof(float), cudaMemcpyHostToDevice));
//    CUDA_CHECK(cudaMemset(result_d, 0, size * sizeof(float)));
    cudaMalloc((void**)&array2_d, size * sizeof(float));
    cudaMalloc((void**)&result_d, size * sizeof(float));
    cudaMemcpy(array1_d, array1_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array2_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(result_d, 0, size * sizeof(float));

    for(int j = 0; j < loop_times; j++) {
        ArrayAddV2<<<(size + 255) / 256, 256>>>(array1_d, array2_d, result_d);
    }
    cudaDeviceSynchronize();
//    delete[] array1_h;
//    delete[] array2_h;
//    delete[] result_h;
//    CUDA_CHECK(cudaFree(array1_d));
//    CUDA_CHECK(cudaFree(array2_d));
//    CUDA_CHECK(cudaFree(result_d));
    return 0;
}