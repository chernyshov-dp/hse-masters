#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 16

__global__ void matrix_multiplication(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    double *hA, *hB, *hC;

    // Using Unified Memory
    cudaMallocManaged(&hA, N * N * sizeof(double));
    cudaMallocManaged(&hB, N * N * sizeof(double));
    cudaMallocManaged(&hC, N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        hA[i] = (double)rand() / RAND_MAX;
        hB[i] = (double)rand() / RAND_MAX;
    }

    dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    matrix_multiplication<<<gridDim, blockDim>>>(hA, hB, hC, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", hC[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(hA);
    cudaFree(hB);
    cudaFree(hC);

    return 0;
}
