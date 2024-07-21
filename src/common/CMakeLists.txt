cmake_minimum_required(VERSION 3.16)
project(common)
# Add the source files
set(SOURCES timer.cpp
        cuda_macros.h)

# Add CUDA specific source files if USE_CUDA is enabled
# if (USE_CUDA)
#     set(CUDA_SOURCES LatencyTimer.cu)
# endif()

# Create the executable
# add_executable(LatencyTimer ${SOURCES} ${CUDA_SOURCES})

# Link CUDA libraries if USE_CUDA is enabled
# if (USE_CUDA)
#     target_link_libraries(LatencyTimer ${CUDA_LIBRARIES})
# endif()

add_library(common ${SOURCES})
