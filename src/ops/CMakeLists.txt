cmake_minimum_required(VERSION 3.12)
project(ops)
# Add the source files

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})  # Set your desired GPU architecture

# Add CUDA specific source files if USE_CUDA is enabled
# if (USE_CUDA)
#     set(CUDA_SOURCES LatencyTimer.cu)
# endif()

# Create the executable

CUDA_ADD_EXECUTABLE(array_add array_add.cu)
target_link_libraries(array_add ${CUDA_LIBRARIES} common)

CUDA_ADD_EXECUTABLE(test test.cu)
target_link_libraries(test ${CUDA_LIBRARIES} common)