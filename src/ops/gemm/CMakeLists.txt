cmake_minimum_required(VERSION 3.12)
# Add the source files

#set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})  # Set your desired GPU architecture

# Add CUDA specific source files if USE_CUDA is enabled
# if (USE_CUDA)
#     set(CUDA_SOURCES LatencyTimer.cu)
# endif()

# Create the executable
find_package(benchmark)
find_package(pthread)

CUDA_ADD_EXECUTABLE(test_gemm 
    gemm_cpu.cpp
    gemm_cuda_v0.cu
    gemm_cuda_v1.cu
    benchmark_gemm.cpp
    )

target_link_libraries(test_gemm ${CUDA_LIBRARIES} common benchmark::benchmark pthread)
