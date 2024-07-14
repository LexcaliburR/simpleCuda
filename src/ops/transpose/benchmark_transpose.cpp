#include <benchmark/benchmark.h>

#include "transpose.h"


static void BM_TransposeCPU(benchmark::State& state) {
    std::vector<int> src_shape = {(int)state.range(0), (int)state.range(1), (int)state.range(2), (int)state.range(3)};
    std::vector<int> dist_axis = {(int)state.range(4), (int)state.range(5), (int)state.range(6), (int)state.range(7)};
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }

    float* A = new float[cnt];
    float* B = new float[cnt];
    for(size_t i = 0; i < cnt; i++) {
        A[i] = i;
        B[i] = 0;
    }
    for(auto _ : state) {
        TransposeCPU(A, B, src_shape, dist_axis);
    }
    delete[] A;
    delete[] B;
}


BENCHMARK(BM_TransposeCPU)->Args({{6, 256, 32, 128, 2, 0, 1, 3}})->Unit(benchmark::kMillisecond);

static void BM_TransposeCPUOPENMP(benchmark::State& state) {
    std::vector<int> src_shape = {(int)state.range(0), (int)state.range(1), (int)state.range(2), (int)state.range(3)};
    std::vector<int> dist_axis = {(int)state.range(4), (int)state.range(5), (int)state.range(6), (int)state.range(7)};
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }

    float* A = new float[cnt];
    float* B = new float[cnt];
    for(size_t i = 0; i < cnt; i++) {
        A[i] = i;
        B[i] = 0;
    }
    for(auto _ : state) {
        TransposeCPUOPENMP(A, B, src_shape, dist_axis);
    }
    delete[] A;
    delete[] B;
}
BENCHMARK(BM_TransposeCPUOPENMP)->Args({{6, 256, 32, 128, 2, 0, 1, 3}})->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
