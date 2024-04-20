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
    double *dA, *dB, *dC;

    int nStreams = 4;
    int size = N * N / nStreams;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Using Pinned Memory
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
    
    dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }
    
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < nStreams; i++) {
        cudaMemcpyAsync(dA + i * size, hA + i * size, sizeof(double) * size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(dB + i * size, hB + i * size, sizeof(double) * size, cudaMemcpyHostToDevice, stream[i]);
        
        matrix_multiplication<<<gridDim, blockDim, 0, stream[i]>>>(dA + i * size, dB + i * size, dC + i * size, N);
        
        cudaMemcpyAsync(hC + i * size, dC + i * size, sizeof(double) * size, cudaMemcpyDeviceToHost, stream[i]);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Elapsed time: %.3f ms\n", elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
        printf("%f ", hC[j * N + i]);
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