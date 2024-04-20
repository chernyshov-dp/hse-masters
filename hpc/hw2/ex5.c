#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

void matrix_multiplication(double *A, double *B, double *C, int n) {
    #pragma omp target teams distribute parallel for map(from:C[0:n*n]) map(to:A[0:n*n], B[0:n*n])
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += A[i * n + row] * B[col * n + i];
            }
            C[col * n + row] = sum;
        }
    }
}

int main() {
    double *hA, *hB, *hC;

    hA = (double*)malloc(N * N * sizeof(double));
    hB = (double*)malloc(N * N * sizeof(double));
    hC = (double*)malloc(N * N * sizeof(double));
    
    for (int i = 0; i < N * N; i++) {
        hA[i] = (double)rand() / RAND_MAX;
        hB[i] = (double)rand() / RAND_MAX;
    }
    
    matrix_multiplication(hA, hB, hC, N);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", hC[j * N + i]);
        }
        printf("\n");
    }
    
    free(hA);
    free(hB);
    free(hC);
    
    return 0;
}
