

cmake_minimum_required(VERSION 3.16)
# Add the source files
project(transpose LANGUAGES CXX CUDA)

#set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})  # Set your desired GPU architecture
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Create the executable
find_package(benchmark)
# find_package(pthread)
find_package(CUDA REQUIRED)
find_library(CUDA_CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
list(APPEND CUDA_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})


# globla include directories
include_directories(${BENCHMARK_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(bm_transpose
        transpose_cpu.cpp
        transpose_cuda.cu
        bm_transpose.cpp
)
target_link_libraries(bm_transpose ${CUDA_LIBRARIES} common benchmark::benchmark pthread)
target_include_directories(bm_transpose PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bm_cuda_memcpy bm_cuda_memcpy.cpp)
target_link_libraries(bm_cuda_memcpy ${CUDA_LIBRARIES} common benchmark::benchmark)
target_include_directories(bm_cuda_memcpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
