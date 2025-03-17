#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 4
#define MAX_VALUE 100

int main() {
    int values[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // Initialize random seed
    srand(time(NULL));
    
    // Generate values in parallel
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        int local_value = rand() % MAX_VALUE + 1;
        
        #pragma omp critical
        {
            values[tid] = local_value;
            thread_ids[tid] = tid;
            printf("Thread %d generated value: %d (unsorted)\n", tid, local_value);
        }
    }
    
    // Sort values (bubble sort for simplicity)
    for (int i = 0; i < NUM_THREADS - 1; i++) {
        for (int j = 0; j < NUM_THREADS - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                // Swap values
                int temp_val = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp_val;
                
                // Swap thread IDs too
                int temp_id = thread_ids[j];
                thread_ids[j] = thread_ids[j + 1];
                thread_ids[j + 1] = temp_id;
            }
        }
    }
    
    printf("\nSorted output:\n");
    // Print in sorted order
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("Thread %d generated value: %d\n", thread_ids[i], values[i]);
    }
    
    return 0;
}
