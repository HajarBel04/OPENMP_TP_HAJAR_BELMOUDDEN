#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N 1000  // Matrix dimensions

int main() {
    int i, j;
    double sum_parallel = 0.0;
    double sum_serial = 0.0;
    double start_time, end_time;
    
    // Dynamically allocate a 2D matrix.
    double **matrix = malloc(N * sizeof(double *));
    if (matrix == NULL) {
        perror("Failed to allocate memory for matrix rows");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < N; i++) {
        matrix[i] = malloc(N * sizeof(double));
        if (matrix[i] == NULL) {
            perror("Failed to allocate memory for matrix columns");
            exit(EXIT_FAILURE);
        }
    }
    
    // Master thread initializes the matrix.
    #pragma omp parallel
    {
        #pragma omp master
        {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    // Initialize with a value, for instance, using a function of i and j.
                    matrix[i][j] = (double)(i * N + j);
                }
            }
        }
    }
    
    // Use a single thread to print the matrix.
    #pragma omp parallel
    {
        #pragma omp single
        {
            int max_print = (N > 10) ? 10 : N;
            printf("Matrix (first %d rows):\n", max_print);
            for (i = 0; i < max_print; i++) {
                for (j = 0; j < N; j++) {
                    printf("%.2f ", matrix[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    // --- Parallel Sum using OpenMP ---
    start_time = omp_get_wtime();
    #pragma omp parallel for private(j) reduction(+:sum_parallel)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum_parallel += matrix[i][j];
        }
    }
    end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    
    // --- Serial Sum without OpenMP ---
    start_time = omp_get_wtime();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum_serial += matrix[i][j];
        }
    }
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    
    // Print the computed sums and their corresponding execution times.
    printf("Parallel sum: %f, time taken: %f seconds\n", sum_parallel, parallel_time);
    printf("Serial   sum: %f, time taken: %f seconds\n", sum_serial, serial_time);
    
    // Clean up dynamically allocated memory.
    for (i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return 0;
}

