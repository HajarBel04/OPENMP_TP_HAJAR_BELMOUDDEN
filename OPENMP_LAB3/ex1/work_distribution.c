#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define N 1000000  // Array size

int main() {
    // Allocate memory for the array
    double *array = malloc(N * sizeof(double));
    if (array == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator and initialize the array with random values between 0 and 1
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = rand() / (double) RAND_MAX;
    }

    double sum = 0.0;
    double max_val = array[0];
    double mean = 0.0;
    double stddev = 0.0;

    // Record the starting time
    double start_time = omp_get_wtime();

    // Parallel sections for concurrent execution of different tasks
    #pragma omp parallel sections
    {
        // Section 1: Compute the sum of all elements
        #pragma omp section
        {
            double local_sum = 0.0;
            for (int i = 0; i < N; i++) {
                local_sum += array[i];
            }
            sum = local_sum;
        }

        // Section 2: Compute the maximum value
        #pragma omp section
        {
            double local_max = array[0];
            for (int i = 0; i < N; i++) {
                if (array[i] > local_max) {
                    local_max = array[i];
                }
            }
            max_val = local_max;
        }

        // Section 3: Compute the standard deviation
        #pragma omp section
        {
            double local_sum = 0.0;
            // First compute the mean
            for (int i = 0; i < N; i++) {
                local_sum += array[i];
            }
            mean = local_sum / N;

            double variance = 0.0;
            for (int i = 0; i < N; i++) {
                variance += (array[i] - mean) * (array[i] - mean);
            }
            stddev = sqrt(variance / N);
        }
    }

    // Record the ending time
    double end_time = omp_get_wtime();

    // Output the results
    printf("Sum = %f\n", sum);
    printf("Maximum value = %f\n", max_val);
    printf("Standard Deviation = %f\n", stddev);
    printf("Time taken = %f seconds\n", end_time - start_time);

    free(array);
    return 0;
}

