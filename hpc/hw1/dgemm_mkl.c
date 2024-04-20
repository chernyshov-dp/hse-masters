#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mkl.h>


void fill_matrix(double *matrix, int rows, int cols) {
    // Filling matrix with random doubles from 0.0 to 10.0
    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[j * rows + i] = (double)rand() / RAND_MAX * 10.0;
        }
    }
}

int main() {
    int M, N, K;
    double alpha = 1.0, beta = 0.0;
    double start_t, end_t, execution_t;
    // Iterating through sizes of matrixes
    for (int S = 500; S < 1501; S += 500) {
        // Making our matrix square
        M = N = K = S;

        double *A = (double *)malloc(M * K * sizeof(double));
        double *B = (double *)malloc(K * N * sizeof(double));
        double *C = (double *)malloc(M * N * sizeof(double));

        fill_matrix(A, M, K);
        fill_matrix(B, K, N);
        start_t = omp_get_wtime();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
        // Calculating the DGEMM time in seconds
        end_t = omp_get_wtime();
        execution_t = end_t - start_t;
        printf("Elapsed time (N = %d): %.2lf sec\n", S, execution_t);

        free(A);
        free(B);
        free(C);
    }
    return 0;
}
