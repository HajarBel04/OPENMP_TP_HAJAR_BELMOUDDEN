#include <stdio.h>
#include <omp.h>

#define LIGHT_ITERS    50e6    // Task A (light computation)
#define MODERATE_ITERS 100e6   // Task B (moderate computation)
#define HEAVY_ITERS    200e6   // Task C (heavy computation)

// Dummy workload functions simulating different computational loads.
void taskA() {
    volatile double dummy = 0.0;
    long iterations = (long)LIGHT_ITERS;
    for (long i = 0; i < iterations; i++) {
        dummy += 1.0; // Prevent compiler optimizing away the loop.
    }
}

void taskB() {
    volatile double dummy = 0.0;
    long iterations = (long)MODERATE_ITERS;
    for (long i = 0; i < iterations; i++) {
        dummy += 1.0;
    }
}

void taskC() {
    volatile double dummy = 0.0;
    long iterations = (long)HEAVY_ITERS;
    for (long i = 0; i < iterations; i++) {
        dummy += 1.0;
    }
}

// Optimized version: Split heavy task C into two parts.
void taskC_part() {
    volatile double dummy = 0.0;
    long iterations = (long)HEAVY_ITERS / 2;  // Each part does half the work.
    for (long i = 0; i < iterations; i++) {
        dummy += 1.0;
    }
}

// Run the unoptimized version with three parallel sections.
void run_unoptimized() {
    double start, end;
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            taskA();
            printf("Task A (light) completed.\n");
        }
        #pragma omp section
        {
            taskB();
            printf("Task B (moderate) completed.\n");
        }
        #pragma omp section
        {
            taskC();
            printf("Task C (heavy) completed.\n");
        }
    }
    end = omp_get_wtime();
    printf("Unoptimized total execution time: %f seconds\n\n", end - start);
}

// Run the optimized version where heavy task C is split between two sections.
void run_optimized() {
    double start, end;
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            taskA();
            printf("Task A (light) completed.\n");
        }
        #pragma omp section
        {
            taskB();
            printf("Task B (moderate) completed.\n");
        }
        #pragma omp section
        {
            taskC_part();
            printf("Task C Part 1 (heavy, half workload) completed.\n");
        }
        #pragma omp section
        {
            taskC_part();
            printf("Task C Part 2 (heavy, half workload) completed.\n");
        }
    }
    end = omp_get_wtime();
    printf("Optimized total execution time: %f seconds\n\n", end - start);
}

int main() {
    printf("Running unoptimized version (3 sections)...\n");
    run_unoptimized();
    
    printf("Running optimized version (4 sections: split heavy task)...\n");
    run_optimized();
    
    return 0;
}

