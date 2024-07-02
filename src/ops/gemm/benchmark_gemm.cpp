#include <cstdlib> // for rand(), RAND_MAX
#include <cstdio>
#include <benchmark/benchmark.h>

#include "gemm.h"
#include "common/check_utils.h"
#include "common/timer.h"

// test1: 512 x 512
// test2: 1024 x 1024
// test3: 2048 x 2048
// test4: 4096 x 4096

static void BM_GemmCPU(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);
    // const int M = 512;
    // const int N = 512;
    // const int K = 512;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_ref = new float[M * N];
    float alpha = 1.0f;
    float beta = 0.0f;

    // Initialize A, B, C, with same rand value in everytimes
    srand(0);
    for(int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    for(auto _ : state) {
        GemmCPU(A, B, C_ref, M, N, K, false, false, alpha, beta);
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
}
BENCHMARK(BM_GemmCPU)->Args({512, 512, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GemmCPU)->Args({1024, 1024, 1024})->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_GemmCPU)->Args({2048, 2048, 2048})->Unit(benchmark::kMillisecond);
// too slow
// BENCHMARK(BM_GemmCPU)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);


static void BM_GemmGPUV0(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);
    // const int M = 512;
    // const int N = 512;
    // const int K = 512;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_ref = new float[M * N];
    float alpha = 1.0f;
    float beta = 0.0f;
    // Initialize A, B, C, with same rand value in everytimes
    srand(0);
    for(int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for(int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* alpha_d = nullptr;
    float* beta_d = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&alpha_d, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&beta_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(alpha_d, &alpha, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(beta_d, &beta, sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    for(auto _ : state) {
        cudaDeviceSynchronize();
        cudaError_t err = GemmGPU_V0(A, B, C_ref, M, N, K, false, false, alpha, beta);
        if (err != cudaSuccess) {
            state.SkipWithError("cudaMemcpy failed");
            break;
        }
        cudaDeviceSynchronize();
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(alpha_d);
    cudaFree(beta_d);
}

BENCHMARK(BM_GemmGPUV0)->Args({512, 512, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GemmGPUV0)->Args({1024, 1024, 1024})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GemmGPUV0)->Args({2048, 2048, 2048})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GemmGPUV0)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
// int main(int argc, char** argv)
// {

//     Test1();
//     return 0;
// }