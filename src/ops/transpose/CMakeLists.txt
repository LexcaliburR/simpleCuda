

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
# find_package(pthread)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_library(CUDA_CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
list(APPEND CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})

CUDA_ADD_EXECUTABLE(test_transpose 
transpose_cpu.cpp
transpose_cuda.cu
benchmark_transpose.cpp
    )

target_link_libraries(test_transpose ${CUDA_LIBRARIES} common benchmark::benchmark pthread)
