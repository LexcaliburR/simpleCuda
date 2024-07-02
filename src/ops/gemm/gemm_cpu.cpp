

#include "gemm.h"


// C = alpha * A * B + beta * C
// A is M x K
// B is K x N
// C is M x N
void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K, bool TransA, bool TransB, float alpha, float beta)
{
    // C = alpha * A * B + beta * C
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0;
            for(int k = 0; k < K; k++) {
                float a = TransA ? A[k * M + i] : A[i * K + k];
                float b = TransB ? B[j * K + k] : B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}