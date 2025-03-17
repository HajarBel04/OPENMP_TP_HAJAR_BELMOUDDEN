#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1000000

// Fills an array with random values between 0 and 99.
void fill_rand(int n, double *A) {
    for (int i = 0; i < n; i++)
        A[i] = rand() % 100;
}

// Computes the sum of an array.
double Sum_array(int n, double *A) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += A[i];
    return sum;
}

int main() {
    double *A, sum;
    double runtime;
    int flag = 0;  // Synchronization flag (0: not ready, 1: data ready)
    
    A = (double *) malloc(N * sizeof(double));
    if (A == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    
    runtime = omp_get_wtime();
    
    // Use parallel sections: one section for the producer and one for the consumer.
    #pragma omp parallel sections shared(flag, A, sum)
    {
        // Producer section: fills the array.
        #pragma omp section
        {
            fill_rand(N, A);
            // Ensure that writes to A are visible before updating flag.
            #pragma omp flush(A, flag)
            flag = 1;
            #pragma omp flush(flag)
        }
        
        // Consumer section: waits until the array is filled, then computes the sum.
        #pragma omp section
        {
            int local_flag = 0;
            // Busy-wait loop until the producer signals that data is ready.
            while (!local_flag) {
                #pragma omp flush(flag)
                local_flag = flag;
            }
            // Now that the flag is set, compute the sum.
            sum = Sum_array(N, A);
        }
    }
    
    runtime = omp_get_wtime() - runtime;
    printf("In %lf seconds, the sum is %lf\n", runtime, sum);
    free(A);
    
    return 0;
}

