#include <stdio.h>
#include <omp.h>  // (ensure this is included for omp_get_wtime(), if desired)
static long num_steps = 100000;
double step;
int main ()
{
    int i; 
    double x, pi, sum = 0.0;
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel for reduction(+:sum)  // Added line
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("pi = %f\n", pi);
    return 0;
}

