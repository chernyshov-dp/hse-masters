#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>

#define N 1024

int main() {
    double *hA, *hB, *hC;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMallocManaged(&hA, N * N * sizeof(double));
    cudaMallocManaged(&hB, N * N * sizeof(double));
    cudaMallocManaged(&hC, N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        hA[i] = (double)rand() / RAND_MAX;
        hB[i] = (double)rand() / RAND_MAX;
    }

    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, hA, N, hB, N, &beta, hC, N);

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

    cublasDestroy(handle);

    return 0;
}
