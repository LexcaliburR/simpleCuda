//
// Created by lishiqi on 24-7-20.
//

#include <iostream>
#include <vector>
#include <benchmark/benchmark.h>

#include "transpose.h"
#include "common/cuda_macros.h"
#include "common/common.h"

void test1(std::vector<int> f_srcShape, std::vector<int> f_perm)
{
    simplecuda::LatencyTimer timer("cudaMemCpy", false, false);
    size_t cnt = 1;
    for (auto dim : f_srcShape) {
        cnt *= dim;
    }

    float* A = new float[cnt];
    float* B = new float[cnt];
    float* A_out = new float[cnt];
    float* B_out = new float[cnt];

    for (size_t i = 0; i < cnt; i++) {
        A[i] = i;
        B[i] = 0;
    }
    float* A_d = nullptr;
    float* B_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, cnt * 4));
    CUDA_CHECK(cudaMalloc(&B_d, cnt * 4));
    PERF_START("MemCpy")

    for (int i = 0; i < 100; i++) {
        timer.Start();
        CUDA_CHECK(cudaMemcpy(A_d, A, cnt * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_d, B, cnt * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(A_out, A_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B_out, B_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        PERF_END
        timer.Toc();
    }
    cudaDeviceSynchronize();
    timer.Print();
    simplecuda::PrintArray(A_out, cnt);

    // CUDA_CHECK(cudaFree());
}

static void BM_GPUMemCpy(benchmark::State& state)
{
    std::vector<int> f_srcShape = {(int)state.range(0), (int)state.range(1), (int)state.range(2), (int)state.range(3)};
    std::vector<int> dist_axis = {(int)state.range(4), (int)state.range(5), (int)state.range(6), (int)state.range(7)};
    size_t cnt = 1;
    for (auto dim : f_srcShape) {
        cnt *= dim;
    }

    float* A = new float[cnt];
    float* B = new float[cnt];
    float* A_out = new float[cnt];
    float* B_out = new float[cnt];

    for (size_t i = 0; i < cnt; i++) {
        A[i] = i;
        B[i] = 0;
    }
    float* A_d = nullptr;
    float* B_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, cnt * 4));
    CUDA_CHECK(cudaMalloc(&B_d, cnt * 4));

    PERF_START("MemCpy")

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(A_d, A, cnt * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_d, B, cnt * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(A_out, A_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B_out, B_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    delete[] A;
    delete[] B;
    delete[] A_out;
    delete[] B_out;
}

BENCHMARK(BM_GPUMemCpy)
    ->Args({{6, 256, 24, 16, 1, 0, 2, 3}})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Threads(1);

BENCHMARK_MAIN();

// int main()
// {
//     std::cout << "------ transpose test start -------" << std::endl;
//     test1({6, 256, 24, 16}, {1, 0, 2, 3});
//     std::cout << "------ transpose test end -------" << std::endl;
// }