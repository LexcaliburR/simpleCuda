#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>

#include "transpose.h"
#include "common/cuda_macros.h"

static void BM_TransposeCPU(benchmark::State& state)
{
    std::vector<int> src_shape = {static_cast<int>(state.range(0)),
                                  static_cast<int>(state.range(1)),
                                  static_cast<int>(state.range(2)),
                                  static_cast<int>(state.range(3))};
    std::vector<int> dist_axis = {static_cast<int>(state.range(4)),
                                  static_cast<int>(state.range(5)),
                                  static_cast<int>(state.range(6)),
                                  static_cast<int>(state.range(7))};
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }

    auto const A = new float[cnt];
    auto const B = new float[cnt];
    for (size_t i = 0; i < cnt; i++) {
        A[i] = static_cast<float>(i);
        B[i] = 0;
    }
    for (auto _ : state) {
        TransposeCPU(A, B, src_shape, dist_axis);
    }
    delete[] A;
    delete[] B;
}

BENCHMARK(BM_TransposeCPU)->Args({{6, 256, 24, 16, 2, 0, 1, 3}})->Unit(benchmark::kMillisecond);

static void BM_TransposeCPUOPENMP(benchmark::State& state)
{
    std::vector<int> src_shape = {static_cast<int>(state.range(0)),
                                  static_cast<int>(state.range(1)),
                                  static_cast<int>(state.range(2)),
                                  static_cast<int>(state.range(3))};
    std::vector<int> dist_axis = {static_cast<int>(state.range(4)),
                                  static_cast<int>(state.range(5)),
                                  static_cast<int>(state.range(6)),
                                  static_cast<int>(state.range(7))};
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }

    auto const A = new float[cnt];
    auto const B = new float[cnt];
    for (size_t i = 0; i < cnt; i++) {
        A[i] = static_cast<float>(i);
        B[i] = 0;
    }
    for (auto _ : state) {
        TransposeCPUOPENMP(A, B, src_shape, dist_axis);
    }
    delete[] A;
    delete[] B;
}
BENCHMARK(BM_TransposeCPUOPENMP)->Args({{6, 256, 24, 16, 2, 0, 1, 3}})->Unit(benchmark::kMillisecond);

static void BM_TransposeGPUNaive(benchmark::State& state)
{
    std::vector<int> src_shape = {static_cast<int>(state.range(0)),
                                  static_cast<int>(state.range(1)),
                                  static_cast<int>(state.range(2)),
                                  static_cast<int>(state.range(3))};
    std::vector<int> dist_axis = {static_cast<int>(state.range(4)),
                                  static_cast<int>(state.range(5)),
                                  static_cast<int>(state.range(6)),
                                  static_cast<int>(state.range(7))};
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }

    auto const A = new float[cnt];
    auto const B = new float[cnt];
    auto const A_out = new float[cnt];
    auto const B_out = new float[cnt];
    for (size_t i = 0; i < cnt; i++) {
        A[i] = static_cast<float>(i);
        B[i] = 0;
    }
    float* A_d = nullptr;
    float* B_d = nullptr;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&A_d, cnt * 4));
    CUDA_CHECK(cudaMalloc(&B_d, cnt * 4));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A_d, A, cnt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, cnt * sizeof(float), cudaMemcpyHostToDevice));
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();

        CUDA_CHECK(TransposeGPUNaive(A_d, B_d, src_shape, dist_axis));

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    CUDA_CHECK(cudaMemcpy(A_out, A_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_out, B_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));

    delete[] A;
    delete[] B;
    cudaFree(A_d);
    cudaFree(B_d);
    delete[] A_out;
    delete[] B_out;
}
BENCHMARK(BM_TransposeGPUNaive)->Args({{6, 256, 24, 16, 1, 0, 2, 3}})->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_MAIN();
