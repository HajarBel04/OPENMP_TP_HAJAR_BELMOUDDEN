#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000  // Size of the data array

int main() {
    int data[N];
    int processed_data[N];
    int final_result = 0;
    int i;
    
    #pragma omp parallel shared(data, processed_data, final_result) private(i)
    {
        // Stage 1: Only one thread reads (initializes) the input data.
        #pragma omp single
        {
            printf("Stage 1: Reading input data\n");
            for (i = 0; i < N; i++) {
                // Simulate reading data. For example, initialize array with values.
                data[i] = i;
            }
        }
        
        // Barrier: Ensure all threads wait until Stage 1 is finished.
        #pragma omp barrier
        
        // Stage 2: All threads process the data in parallel.
        // For example, multiply each element by 2.
        #pragma omp for
        for (i = 0; i < N; i++) {
            processed_data[i] = data[i] * 2;
        }
        
        // Barrier: Wait until all threads finish processing the data.
        #pragma omp barrier
        
        // Stage 3: Only one thread writes the final result.
        // For example, compute the sum of the processed data.
        #pragma omp single
        {
            printf("Stage 3: Writing final result\n");
            for (i = 0; i < N; i++) {
                final_result += processed_data[i];
            }
            printf("Final result (sum of processed data): %d\n", final_result);
        }
    }
    
    return 0;
}

