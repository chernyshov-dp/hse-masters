#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "time.h"


const double l = 1.0;        // Length of the rod
const double u_0 = 1.0;      // Initial temperature
const double k = 1;          // Thermal conductivity coefficient
const double h = 0.02;       // Spatial step
const double tau = 0.0002;   // Time step
const double T = 0.1;        // Time

double exact_solution(double x, double t)
{
    double sum = 0.0;
    for (int m = 0; m < 1000; ++m)
    {
        sum += exp(-k * M_PI * M_PI * (2 * m + 1) * (2 * m + 1) * t / (l * l)) /
               (2 * m + 1) * sin(M_PI * (2 * m + 1) * x / l);
    }
    return (4 * u_0 / M_PI) * sum;
}

double *next_temp_distribution(double *temp, const int k, const double h, const double tau, const int rank, const int num_proc, const int frags_per_proc)
{
    double left_el, right_el;
    MPI_Status status;

    // Synchronous send and receive to exchange boundary temperatures
    if (rank == 0)
    {
        left_el = 0.0;
        MPI_Sendrecv(temp + frags_per_proc - 1, 1, MPI_DOUBLE, rank + 1, 10,
                     &right_el, 1, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &status);
    }
    else if (rank == num_proc - 1)
    {
        MPI_Sendrecv(temp, 1, MPI_DOUBLE, rank - 1, 11,
                     &left_el, 1, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &status);
        right_el = 0.0;
    }
    else
    {
        MPI_Sendrecv(temp + frags_per_proc - 1, 1, MPI_DOUBLE, rank + 1, 10,
                     &right_el, 1, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(temp, 1, MPI_DOUBLE, rank - 1, 11,
                     &left_el, 1, MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &status);
    }

    // Compute new temperatures using the finite difference scheme
    double *new_temp = (double *)malloc(frags_per_proc * sizeof(double));
    for (int i = 0; i < frags_per_proc; i++)
    {
        double ul, ur, u;
        ul = (i != 0) ? (temp[i - 1]) : (left_el);
        ur = (i != frags_per_proc - 1) ? (temp[i + 1]) : (right_el);
        u = temp[i];

        new_temp[i] = u + k * tau / (h * h) * (ul - 2 * u + ur);
    }

    // Update the original temperature array
    memcpy(temp, new_temp, frags_per_proc * sizeof(double));
    free(new_temp);

    return temp;
}

double get_temp(const double point, double *temp, const double h)
{
    int point_idx = (int)(point / h);
    return temp[point_idx];
}

int main(int argc, char **argv)
{
    // Initialize constants and initial temperature distribution
    int num_frag = (int)(1 / h);
    double *temp = (double *)malloc(num_frag * sizeof(double));
    for (int i = 0; i < num_frag; i++)
        temp[i] = 1;

    int num_proc;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate the number of fragments per process
    int frags_per_proc = num_frag / num_proc;
    double *temp_reg = (double *)malloc(frags_per_proc * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time;
    if (rank == 0)
        start_time = MPI_Wtime();

    // Distribute initial temperature distribution to all processes
    MPI_Scatter(temp, frags_per_proc, MPI_DOUBLE, temp_reg, frags_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate temperature by timestamps
    for (double cur_tau = 0.0; cur_tau < T; cur_tau += tau)
    {
        next_temp_distribution(temp_reg, k, h, tau, rank, num_proc, frags_per_proc);
    }

    // Gather calculated temperatures back to process 0
    double *gathered_temp = NULL;
    if (rank == 0)
    {
        gathered_temp = (double *)malloc(num_frag * sizeof(double));
    }
    MPI_Gather(temp_reg, frags_per_proc, MPI_DOUBLE, gathered_temp, frags_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        double end_time = MPI_Wtime();
        printf("Execution time: %f\n", end_time - start_time);

        for (int point_idx = 0; point_idx <= 10; ++point_idx)
        {
            double point = (double)point_idx * 0.1;
            printf("Point %d: Exact = %f Computed = %f\n", point_idx, exact_solution(point, T), get_temp(point, gathered_temp, h));
        }

        free(gathered_temp);
    }

    free(temp_reg);
    free(temp);
    MPI_Finalize();

    return 0;
}

