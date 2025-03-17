#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    double pi, sum = 0.0;
    step = 1.0 / (double) num_steps;
    
    double start_time = omp_get_wtime();
    
    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        double local_sum = 0.0;
        
        // Declare loop variable 'i' locally so that it is private to each thread.
        for (int i = id; i < num_steps; i += nthrds) {
            double x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }
        
        // Safely add the local sum to the global sum.
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    
    pi = step * sum;
    double end_time = omp_get_wtime();
    
    printf("pi = %f\n", pi);
    printf("Time taken = %f seconds\n", end_time - start_time);
    
    return 0;
}

