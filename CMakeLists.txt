cmake_minimum_required(VERSION 3.12)
project(simplecuda C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add an option to enable/disable CUDA support
option(USE_CUDA "Enable CUDA support" ON)

find_package(CUDA REQUIRED)

# Add CUDA specific configuration if USE_CUDA is enabled
if (USE_CUDA)
    find_package(CUDA REQUIRED)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14)
#     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_86)  # Set your desired GPU architecture
endif()

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--generate-code=arch=compute_86,code=sm_86)  # Set your desired GPU architecture
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--gpu-architecture=all)  # Set your desired GPU architecture
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--generate-code=arch=compute_86,code=sm_86)  # Set your desired GPU architecture

add_definitions(-DUSE_CUDA)
include_directories(${ROOT_DIR}/src ${CUDA_INCLUDE_DIRS})

add_subdirectory(src/common)
add_subdirectory(src/ops)

