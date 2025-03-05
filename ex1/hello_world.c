#include <stdio.h>

#ifdef _OPENMP
  #include <omp.h>
#else
  // Dummy definitions for sequential execution
  int omp_get_thread_num(void) { return 0; }
  int omp_get_num_threads(void) { return 1; }
#endif

int main() {
    int num_threads, rank;

    #ifdef _OPENMP
      #pragma omp parallel private(rank)
      {
          rank = omp_get_thread_num();
          num_threads = omp_get_num_threads();
          printf("Hello from the rank %d thread\n", rank);
      }
    #else
          rank = 0;
          num_threads = 1;
          printf("Hello from the rank %d thread\n", rank);
    #endif

    printf("Parallel execution of hello_world with %d threads\n", num_threads);
    return 0;
}

