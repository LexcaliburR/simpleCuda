
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

std::chrono::high_resolution_clock::time_point getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

double durationToMilliseconds(std::chrono::high_resolution_clock::duration duration) {
    double duration_ms = std::chrono::duration<double>(duration).count() * 1000.0;
    return duration_ms;
}

__global__ void warmup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;

    } else {
        b = 200.0f;
    }
    // printf("%d %d %f \n",tid,warpSize,a+b);
    c[tid] = a + b;
}
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a = 0.0;
    float b = 0.0;
    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    bool ipred = (tid % 2 == 0);
    if (ipred) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    int id_element_per_warp = (tid % 32);
    if (id_element_per_warp == 1) {
        a = 100.0f;
    } else if (id_element_per_warp == 2) {
        b = 200.0f;
    } else if (id_element_per_warp == 3) {
        b = 200.0f;
    } else if (id_element_per_warp == 5) {
        b = 200.0f;
    } else if (id_element_per_warp == 2) {
        b = 200.0f;
    } else if (id_element_per_warp == 9) {
        b = 200.0f;
    } else if (id_element_per_warp == 23) {
        b = 200.0f;
    }  else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 1024000000;
    int blocksize = 1024;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *C_dev;
    size_t nBytes = size * sizeof(float);
    float *C_host = (float *)malloc(nBytes);
    cudaMalloc((float **)&C_dev, nBytes);

    // run a warmup kernel to remove overhead
    auto iStart = getCurrentTime();
    double iElaps = 0;
    
    cudaDeviceSynchronize();
    iStart = getCurrentTime();
    warmup<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = durationToMilliseconds(getCurrentTime() - iStart);
    printf("warmup	  <<<%d,%d>>>elapsed %lf ms \n", grid.x, block.x, iElaps);

    // run kernel 1
    iStart = getCurrentTime();
    mathKernel1<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = durationToMilliseconds(getCurrentTime() - iStart);
    printf(
        "mathKernel1<<<%4d,%4d>>>elapsed %lf ms \n", grid.x, block.x, iElaps);
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost);
    // for(int i=0;i<size;i++)
    //{
    //	printf("%f ",C_host[i]);
    // }
    // run kernel 2
    iStart = getCurrentTime();
    mathKernel2<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = durationToMilliseconds(getCurrentTime() - iStart);
    printf(
        "mathKernel2<<<%4d,%4d>>>elapsed %lf ms \n", grid.x, block.x, iElaps);

    // run kernel 3
    iStart = getCurrentTime();
    mathKernel3<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = durationToMilliseconds(getCurrentTime() - iStart);
    printf(
        "mathKernel3<<<%4d,%4d>>>elapsed %lf ms \n", grid.x, block.x, iElaps);

    iStart = getCurrentTime();
    mathKernel4<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = durationToMilliseconds(getCurrentTime() - iStart);
    printf(
        "mathKernel4<<<%4d,%4d>>>elapsed %lf ms \n", grid.x, block.x, iElaps);

    cudaFree(C_dev);
    free(C_host);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}