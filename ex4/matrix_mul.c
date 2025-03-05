#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    // Define dimensions: m = rows in A and C, n = columns in A and rows in B.
    int m = 1000;  // for example, change as needed
    int n = 1000;  // for example, change as needed

    // Dynamically allocate matrices as 1D arrays
    double *a = (double *) malloc(m * n * sizeof(double));
    double *b = (double *) malloc(n * m * sizeof(double));
    double *c = (double *) malloc(m * m * sizeof(double));

    // Initialize matrix A: A(i,j) = (i+1) + (j+1)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }

    // Initialize matrix B: B(i,j) = (i+1) - (j+1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }

    // Initialize matrix C to zero.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0.0;
        }
    }

    // Get starting time.
    double start_time = omp_get_wtime();

    // Matrix multiplication: C = A x B
    // We use collapse(2) to combine the i and j loops into one large iteration space.
    // Change schedule() clause to test STATIC, DYNAMIC, or GUIDED scheduling and adjust chunk sizes.
    #pragma omp parallel for collapse(2) schedule(static)
    // Example alternatives:
    //#pragma omp parallel for collapse(2) schedule(dynamic, 10)
    //#pragma omp parallel for collapse(2) schedule(guided, 5)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }

    // Get ending time.
    double end_time = omp_get_wtime();

    // Calculate and print the elapsed time.
    printf("Matrix multiplication took %f seconds\n", end_time - start_time);

    // Optionally, verify or print some elements from C.
    printf("C[0,0] = %f\n", c[0]);
    printf("C[m-1, m-1] = %f\n", c[m*m - 1]);

    // Free allocated memory.
    free(a);
    free(b);
    free(c);

    return 0;
}

