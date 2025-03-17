#include <stdio.h>
#include <omp.h>

#define NUM_INCREMENTS 100000000

int main() {
    long counter = 0;
    double start, end;

    // --- Version 1: Using critical section ---
    counter = 0;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        for (long i = 0; i < NUM_INCREMENTS; i++) {
            #pragma omp critical
            {
                counter++;
            }
        }
    }
    end = omp_get_wtime();
    printf("Critical section: Counter = %ld, Time = %f seconds\n", counter, end - start);

    // --- Version 2: Using atomic directive ---
    counter = 0;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        for (long i = 0; i < NUM_INCREMENTS; i++) {
            #pragma omp atomic
            counter++;
        }
    }
    end = omp_get_wtime();
    printf("Atomic directive: Counter = %ld, Time = %f seconds\n", counter, end - start);

    return 0;
}

