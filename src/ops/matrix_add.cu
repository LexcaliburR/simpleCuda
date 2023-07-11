


#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <iostream>
#include "common/common.h"
#include "common/timer.h"

// C = A + B
__global__ void MatrixAddV1(float* matrix_A, float* matrix_B, float* matrix_C, const size_t n_rows, const size_t n_cols) {
    size_t index_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(index_x < n_cols && index_y < n_rows) {
        size_t index_offset = index_y * n_cols + index_x;
        matrix_C[index_offset] = matrix_A[index_offset] + matrix_B[index_offset];
    }
    return;
}

// 2d grid, 2d block
cudaError_t MatrixAddV1Launch1(float* matrix_A, float* matrix_B, float* matrix_C, const size_t n_rows, const size_t n_cols, cudaStream_t stream= nullptr) {
    dim3 block(16, 16);
    dim3 grid(DIVUP(n_cols, 16), DIVUP(n_rows, 16));
    MatrixAddV1<<<grid, block>>>(matrix_A, matrix_B, matrix_C, n_rows, n_cols);
    auto err = cudaGetLastError();
    return err;
}

// 1d grid, 1d block
cudaError_t MatrixAddV1Launch2(float* matrix_A, float* matrix_B, float* matrix_C, const size_t n_rows, const size_t n_cols, cudaStream_t stream= nullptr) {
    dim3 block(32);
    dim3 grid(DIVUP(n_cols * n_rows, 32));
    MatrixAddV1<<<grid, block>>>(matrix_A, matrix_B, matrix_C, n_rows, n_cols);
    auto err = cudaGetLastError();
    return err;
}

// 2d grid, 1d block
cudaError_t MatrixAddV1Launch3(float* matrix_A, float* matrix_B, float* matrix_C, const size_t n_rows, const size_t n_cols, cudaStream_t stream= nullptr) {
    dim3 block(32);
    dim3 grid(DIVUP(n_cols, 32), n_rows);
    MatrixAddV1<<<grid, block>>>(matrix_A, matrix_B, matrix_C, n_rows, n_cols);
    auto err = cudaGetLastError();
    return err;
}

int main()
{
    simplecuda::LatencyTimer cpu_timer("cpu", false, false);
    simplecuda::LatencyTimer addv1_timer1("addv1_1", false, false);
    simplecuda::LatencyTimer addv1_timer2("addv1_2", false, false);
    simplecuda::LatencyTimer addv1_timer3("addv1_3", false, false);

    size_t n_rows = 1000;
    size_t n_cols = 1000;
    size_t n_element = n_cols * n_rows;

    int loop_times = 100;
    float* mat_a_h = new float[n_element];
    float* mat_b_h = new float[n_element];
    float* mat_c_h = new float[n_element];
    for(int i = 0; i < n_rows; ++i) {
        for(int j = 0; j < n_cols; ++j) {
            size_t offset = i * n_rows + j;
            mat_a_h[offset] = 1.1f;
            mat_b_h[offset] = 2.2f;
            mat_c_h[offset] = 0.f;
        }
    }

    float* mat_a_d = nullptr;
    float* mat_b_d = nullptr;
    float* mat_c_d = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&mat_a_d, n_element * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&mat_b_d, n_element * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&mat_c_d, n_element * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(mat_a_d, mat_a_h, n_element * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mat_b_d, mat_b_h, n_element * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(mat_c_d, 0, n_element * sizeof(float)));

    for(int j = 0; j < loop_times; j++) {
        cpu_timer.Start();
        for(int row = 0; row < n_rows; ++row) {
            for(int col = 0; col < n_cols; ++col) {
                size_t offset = row * n_rows + col;
                mat_c_h[offset] = mat_a_h[offset] + mat_b_h[offset];
            }
        }
        cpu_timer.Toc();
    }

    cpu_timer.Print();
    cpu_timer.SaveToFile();

    // warm up
    for(int i = 0; i < 1000; i++) {
        CUDA_CHECK(MatrixAddV1Launch1(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
        CUDA_CHECK(MatrixAddV1Launch2(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
        CUDA_CHECK(MatrixAddV1Launch3(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for(int j = 0; j < loop_times; j++) {
        addv1_timer1.Start();
        CUDA_CHECK(MatrixAddV1Launch1(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
        CUDA_CHECK(cudaDeviceSynchronize());
        addv1_timer1.Toc();
    }
    printf("grid(%d, %d), block(%d, %d)\n", DIVUP(n_cols, 16), DIVUP(n_rows, 16), 16, 16);
    addv1_timer1.Print();
    addv1_timer1.SaveToFile();

    for(int j = 0; j < loop_times; j++) {
        //        addv1_timer.Start(stream_addv1);
        addv1_timer2.Start();
        CUDA_CHECK(MatrixAddV1Launch2(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
        CUDA_CHECK(cudaDeviceSynchronize());
        addv1_timer2.Toc();
        //        addv1_timer.Toc(stream_addv1);
    }
    printf("grid(%d), block(%d)\n", DIVUP(n_cols * n_rows, 32), 32);

    addv1_timer2.Print();
    addv1_timer2.SaveToFile();

    for(int j = 0; j < loop_times; j++) {
        addv1_timer3.Start();
        CUDA_CHECK(MatrixAddV1Launch3(mat_a_d, mat_b_d, mat_c_d, n_rows, n_cols))
        CUDA_CHECK(cudaDeviceSynchronize());
        addv1_timer3.Toc();
    }
    printf("grid(%d, %d), block(%d)\n", DIVUP(n_cols, 32), n_rows , 32);
    addv1_timer3.Print();
    addv1_timer3.SaveToFile();


    delete[] mat_a_h;
    delete[] mat_b_h;
    delete[] mat_c_h;
    CUDA_CHECK(cudaFree(mat_a_d));
    CUDA_CHECK(cudaFree(mat_b_d));
    CUDA_CHECK(cudaFree(mat_c_d));
    cudaDeviceReset();

    return 0;
}