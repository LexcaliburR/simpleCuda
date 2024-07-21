//
// Created by lishiqi on 24-7-20.
//

#pragma once

#include <cuda_runtime.h>

#define ThreadPerBlockDefault 1024

#define DIV(x, y) (x - 1) / y + 1

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }
