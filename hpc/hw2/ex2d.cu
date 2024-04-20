#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024
#define THREADS_PER_BLOCK 16

__global__ void matrix_multiplication(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double sharedA[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ double sharedB[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    
    double sum = 0.0;
    
    // Iterating over submatrices
    for (int k = 0; k < (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; k++) {
        // Calculation of column and row indices of a submatrix
        int subCol = k * THREADS_PER_BLOCK + threadIdx.x;
        int subRow = k * THREADS_PER_BLOCK + threadIdx.y;
        
        if (subRow < n && col < n) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * n + subCol];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (subCol < n && row < n) {
            sharedB[threadIdx.y][threadIdx.x] = B[subRow * n + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    double *hA, *hB, *hC;
    double *dA, *dB, *dC;
    
    cudaMallocHost((void**)&hA, N * N * sizeof(double));
    cudaMallocHost((void**)&hB, N * N * sizeof(double));
    cudaMallocHost((void**)&hC, N * N * sizeof(double));
    
    for (int i = 0; i < N * N; i++) {
        hA[i] = (double)rand() / RAND_MAX;
        hB[i] = (double)rand() / RAND_MAX;
    }
    
    cudaMalloc((void**)&dA, N * N * sizeof(double));
    cudaMalloc((void**)&dB, N * N * sizeof(double));
    cudaMalloc((void**)&dC, N * N * sizeof(double));
    
    cudaMemcpy(dA, hA, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * N * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    
    matrix_multiplication<<<gridDim, blockDim>>>(dA, dB, dC, N);
    
    cudaMemcpy(hC, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", hC[i * N + j]);
        }
        printf("\n");
    }
    
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    
    return 0;
}