#include "transpose.h"


void TransposeCPU(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis) {
    
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }
    std::vector<int> dist_shape;
    for (auto i : axis) {
        dist_shape.push_back(src_shape[i]);
    }

    std::vector<int> src_stride(src_shape.size(), 1);
    std::vector<int> dist_stride(dist_shape.size(), 1);
    for (int i = src_shape.size() - 2; i >= 0; i--) {
        src_stride[i] = src_shape[i + 1] * src_stride[i + 1];
    }
    for (int i = dist_shape.size() - 2; i >= 0; i--) {
        dist_stride[i] = dist_shape[i + 1] * dist_stride[i + 1];
    }

    for (size_t i = 0; i < cnt; i++) {
        int src_idx = 0;
        int dist_idx = 0;
        for (size_t j = 0; j < src_shape.size(); j++) {
            int src_tmp = i / src_stride[j] % src_shape[j];
            for (size_t k = 0; k < axis.size(); k++) {
                if (j == axis[k]) {
                    dist_idx += src_tmp * dist_stride[k];
                }
            }
            src_idx += src_tmp * src_stride[j];
        }
        B[dist_idx] = A[src_idx];
    }    
    return;
}

void TransposeCPUOPENMP(const float* A, float* B, const std::vector<int> src_shape, const std::vector<int> axis) {
    
    size_t cnt = 1;
    for (auto dim : src_shape) {
        cnt *= dim;
    }
    std::vector<int> dist_shape;
    for (auto i : axis) {
        dist_shape.push_back(src_shape[i]);
    }

    std::vector<int> src_stride(src_shape.size(), 1);
    std::vector<int> dist_stride(dist_shape.size(), 1);
    for (int i = src_shape.size() - 2; i >= 0; i--) {
        src_stride[i] = src_shape[i + 1] * src_stride[i + 1];
    }
    for (int i = dist_shape.size() - 2; i >= 0; i--) {
        dist_stride[i] = dist_shape[i + 1] * dist_stride[i + 1];
    }

    #pragma omp parallel for 
    for (size_t i = 0; i < cnt; i++) {
        int src_idx = 0;
        int dist_idx = 0;
        for (size_t j = 0; j < src_shape.size(); j++) {
            int src_tmp = i / src_stride[j] % src_shape[j];
            for (size_t k = 0; k < axis.size(); k++) {
                if (j == axis[k]) {
                    dist_idx += src_tmp * dist_stride[k];
                }
            }
            src_idx += src_tmp * src_stride[j];
        }
        B[dist_idx] = A[src_idx];
    }    
    
    return;
}

