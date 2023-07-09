/*
 * @Author: lexcaliburr li289427380@gmail.com
 * @Date: 2023-07-09 11:27:26
 * @LastEditors: lexcaliburr li289427380@gmail.com
 * @LastEditTime: 2023-07-09 16:18:52
 * @FilePath: /simpleCuda/src/common/check_utils.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                      \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }
