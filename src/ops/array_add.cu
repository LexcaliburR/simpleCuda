#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <iostream>
#include "common/common.h"
#include "common/timer.h"

__global__ void ArrayAddV1(float* array1, float* array2, float* result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    result[idx] = array1[idx] + array2[idx];
    return;
}

cudaError_t ArrayAddV1Launch(float* array1, float* array2, size_t size, float* result, cudaStream_t stream= nullptr) {
    ArrayAddV1<<<DIVUP(size, 512), 512>>>(array1, array2, result);
    // ArrayAddV1<<<1, 1>>>(array1, array2, result);

    auto err = cudaGetLastError();
    return err;
}

int main()
{
    simplecuda::LatencyTimer cpu_timer("cpu", false, false);
    simplecuda::LatencyTimer addv1_timer("addv1", false, false);

    size_t size = 100000;
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

    CUDA_CHECK(cudaMalloc((void**)&array1_d, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&array2_d, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&result_d, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(array1_d, array1_h, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(array2_d, array2_h, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(result_d, 0, size * sizeof(float)));

    for(int j = 0; j < loop_times; j++) {
        cpu_timer.Start();
        for(int i = 0; i < size; ++i) {
            result_h[i] = array1_h[i] + array2_h[i];
        }
        cpu_timer.Toc();
    }

    // warm up
    for(int i = 0; i < 4; i++) {
        CUDA_CHECK(ArrayAddV1Launch(array1_d, array2_d, size, result_d));
    }

    cudaStream_t stream_addv1;
    CUDA_CHECK(cudaStreamCreate(&stream_addv1));

    for(int j = 0; j < 4; j++) {
//        addv1_timer.Start(stream_addv1);
        addv1_timer.Start();
        CUDA_CHECK(ArrayAddV1Launch(array1_d, array2_d, size, result_d))
        CUDA_CHECK(cudaDeviceSynchronize());
        addv1_timer.Toc();
//        addv1_timer.Toc(stream_addv1);
    }
    CUDA_CHECK(cudaStreamDestroy(stream_addv1));

    delete[] array1_h;
    delete[] array2_h;
    delete[] result_h;
    CUDA_CHECK(cudaFree(array1_d));
    CUDA_CHECK(cudaFree(array2_d));
    CUDA_CHECK(cudaFree(result_d));
    cudaDeviceReset();

    cpu_timer.Print();
    cpu_timer.SaveToFile();
    addv1_timer.Print();
    addv1_timer.SaveToFile();
    return 0;
}