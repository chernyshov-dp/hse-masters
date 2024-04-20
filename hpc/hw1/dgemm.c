#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


void dgemm(int M, int N, int K, double *A, double *B, double *C) {
    /*
    * M - Number of rows in matrices A and C
    * N - Number of columns in matrices B and C
    * K - Number of columns in matrix A; number of rows in matrix B
    */
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + j] * B[i * K + k];
            }
            C[i * M + j] = sum;
        }
    }
}

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
    double start_t, end_t, execution_t;
    FILE* file = fopen("data.csv", "w");
    fprintf(file, "threads;size;time\n");
    // Iterating through number of threads (T)
    for (int T = 1; T < 17; T *= 2) {
        omp_set_num_threads(T);
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
            dgemm(M, N, K, A, B, C);
            end_t = omp_get_wtime();
            // Calculating the DGEMM time in seconds
            execution_t = end_t - start_t;
            
            // Saving statistics 
            fprintf(file, "%d;%d;%.2lf\n", T, S, execution_t);

            free(A);
            free(B);
            free(C);
        }
    }
    fclose(file);
    return 0;
}
