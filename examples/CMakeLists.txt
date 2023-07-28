cmake_minimum_required(VERSION 3.0.0)
project(example)



CUDA_ADD_EXECUTABLE(1_branch_efficiency 1_branch_efficiency/branch_efficiency.cu)
target_link_libraries(1_branch_efficiency ${CUDA_LIBRARIES})
target_compile_options(1_branch_efficiency PRIVATE -g -G -O0 --device-debug --disable-warnings -code=sm_30 -arch=sm_30)