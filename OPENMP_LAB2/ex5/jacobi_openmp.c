#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef VAL_N
#define VAL_N 120
#endif
#ifndef VAL_D
#define VAL_D 80
#endif

// Random initialization of an array
void random_number(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / (double)(RAND_MAX - 1);
    }
}

int main() {
    int n = VAL_N, diag = VAL_D;
    int i, j, iteration = 0;
    double norme;
    
    // Correct 2D matrix allocation (stored as 1D arrays)
    double *a = (double *) malloc(n * n * sizeof(double));
    double *x = (double *) malloc(n * sizeof(double));
    double *x_courant = (double *) malloc(n * sizeof(double));
    double *b = (double *) malloc(n * sizeof(double));
    if (!a || !x || !x_courant || !b) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Time measurement variables
    struct timeval t_elapsed_0, t_elapsed_1;
    double t_elapsed;
    double t_cpu_0, t_cpu_1, t_cpu;
    
    // Matrix and RHS initialization
    srand(421); // For reproducibility
    random_number(a, n * n);
    random_number(b, n);
    
    // Strengthening the diagonal
    for (i = 0; i < n; i++) {
        a[i * n + i] += diag;
    }
    
    // Initial solution guess
    for (i = 0; i < n; i++) {
        x[i] = 1.0;
    }
    
    // Start timing
    t_cpu_0 = omp_get_wtime();
    gettimeofday(&t_elapsed_0, NULL);
    
    // Jacobi Iteration
    while (1) {
        iteration++;
        
        // Parallelized loop to compute new solution values.
        // Each iteration for a given i is independent (uses the old x)
        #pragma omp parallel for default(none) shared(n, a, b, x, x_courant) private(i, j)
        for (i = 0; i < n; i++) {
            double temp = 0.0;
            // Sum for j < i
            for (j = 0; j < i; j++) {
                temp += a[j * n + i] * x[j];
            }
            // Sum for j > i
            for (j = i + 1; j < n; j++) {
                temp += a[j * n + i] * x[j];
            }
            x_courant[i] = (b[i] - temp) / a[i * n + i];
        }
        
        // Convergence test: compute the maximum absolute difference
        double absmax = 0.0;
        #pragma omp parallel for reduction(max:absmax) default(none) shared(n, x, x_courant)
        for (i = 0; i < n; i++) {
            double curr = fabs(x[i] - x_courant[i]);
            if (curr > absmax)
                absmax = curr;
        }
        norme = absmax / n;
        if ((norme <= DBL_EPSILON) || (iteration >= n))
            break;
        
        // Update x with the newly computed values
        memcpy(x, x_courant, n * sizeof(double));
    }
    
    // End timing
    gettimeofday(&t_elapsed_1, NULL);
    t_elapsed = (t_elapsed_1.tv_sec - t_elapsed_0.tv_sec) +
                (t_elapsed_1.tv_usec - t_elapsed_0.tv_usec) / 1e6;
    t_cpu_1 = omp_get_wtime();
    t_cpu = t_cpu_1 - t_cpu_0;
    
    // Print result
    fprintf(stdout, "\n\n"
                    "         System size           : %5d\n"
                    "         Iterations            : %4d\n"
                    "         Norme                 : %10.3E\n"
                    "         Elapsed time          : %10.3E sec.\n"
                    "         CPU time              : %10.3E sec.\n",
                    n, iteration, norme, t_elapsed, t_cpu);
    
    // Free allocated memory
    free(a);
    free(x);
    free(x_courant);
    free(b);
    
    return EXIT_SUCCESS;
}

