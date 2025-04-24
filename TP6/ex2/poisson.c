#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "compute.h"

// Global variables (defined here, extern in compute.c)
int sx, ex, sy, ey, ntx, nty;

// Function to exchange ghost cells with neighbors
void exchange_ghosts(double *u, int north, int south, int west, int east, MPI_Comm cart_comm) {
    MPI_Status status;
    int local_nx = ex - sx + 1;
    int local_ny = ey - sy + 1;
    
    // Exchange with north/south neighbors
    if (north != MPI_PROC_NULL) {
        // Send top row to north, receive from north into ghost row
        for (int j = sy; j <= ey; j++) {
            double send_val = u[(sx - (sx-1)) * (ey-sy+3) + (j - (sy-1))];
            double recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, north, 0,
                         &recv_val, 1, MPI_DOUBLE, north, 1,
                         cart_comm, &status);
            u[(sx-1 - (sx-1)) * (ey-sy+3) + (j - (sy-1))] = recv_val;
        }
    }
    
    if (south != MPI_PROC_NULL) {
        // Send bottom row to south, receive from south into ghost row
        for (int j = sy; j <= ey; j++) {
            double send_val = u[(ex - (sx-1)) * (ey-sy+3) + (j - (sy-1))];
            double recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, south, 1,
                         &recv_val, 1, MPI_DOUBLE, south, 0,
                         cart_comm, &status);
            u[(ex+1 - (sx-1)) * (ey-sy+3) + (j - (sy-1))] = recv_val;
        }
    }
    
    // Exchange with west/east neighbors
    if (west != MPI_PROC_NULL) {
        // Send left column to west, receive from west into ghost column
        for (int i = sx; i <= ex; i++) {
            double send_val = u[(i - (sx-1)) * (ey-sy+3) + (sy - (sy-1))];
            double recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, west, 2,
                         &recv_val, 1, MPI_DOUBLE, west, 3,
                         cart_comm, &status);
            u[(i - (sx-1)) * (ey-sy+3) + (sy-1 - (sy-1))] = recv_val;
        }
    }
    
    if (east != MPI_PROC_NULL) {
        // Send right column to east, receive from east into ghost column
        for (int i = sx; i <= ex; i++) {
            double send_val = u[(i - (sx-1)) * (ey-sy+3) + (ey - (sy-1))];
            double recv_val;
            MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, east, 3,
                         &recv_val, 1, MPI_DOUBLE, east, 2,
                         cart_comm, &status);
            u[(i - (sx-1)) * (ey-sy+3) + (ey+1 - (sy-1))] = recv_val;
        }
    }
}

// Function to compute global error
double compute_global_error(double *u, double *u_new, MPI_Comm cart_comm) {
    double local_error = 0.0;
    
    for (int i = sx; i <= ex; i++) {
        for (int j = sy; j <= ey; j++) {
            int idx = (i - (sx-1)) * (ey-sy+3) + (j - (sy-1));
            double diff = fabs(u_new[idx] - u[idx]);
            if (diff > local_error) {
                local_error = diff;
            }
        }
    }
    
    double global_error;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    return global_error;
}

int main(int argc, char **argv) {
    int rank, size;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};  // Non-periodic boundaries
    int reorder = 0;
    int coords[2];
    int north, south, east, west;
    
    // Domain parameters
    ntx = 12;  // Number of interior points in x-direction
    nty = 10;  // Number of interior points in y-direction
    
    // Solver parameters
    int max_iterations = 1000;
    double tolerance = 1e-6;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create 2D Cartesian communicator
    MPI_Dims_create(size, 2, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get process coordinates
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Find neighbors
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    
    // Calculate local domain boundaries (using professor's approach)
    sx = (coords[0] * ntx) / dims[0] + 1;
    ex = ((coords[0] + 1) * ntx) / dims[0];
    sy = (coords[1] * nty) / dims[1] + 1;
    ey = ((coords[1] + 1) * nty) / dims[1];
    
    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", ntx, nty);
        printf("Topology dimensions: %d along x, %d along y\n", dims[0], dims[1]);
        printf("-----------------------------------------\n");
    }
    
    printf("Rank in the topology: %d\n", rank);
    printf("Array indices: x from %d to %d, y from %d to %d\n", sx, ex, sy, ey);
    printf("Process %d has neighbors: N %d E %d S %d W %d\n", 
           rank, north, east, south, west);
    
    // Allocate and initialize arrays
    double *u, *u_new, *u_exact;
    initialization(&u, &u_new, &u_exact);
    
    // Start timing
    double start_time = MPI_Wtime();
    
    // Main Jacobi iteration loop
    int iteration;
    for (iteration = 0; iteration < max_iterations; iteration++) {
        // Exchange ghost cells
        exchange_ghosts(u, north, south, west, east, cart_comm);
        
        // Apply Jacobi iteration
        compute(u, u_new);
        
        // Compute global error
        double global_error = compute_global_error(u, u_new, cart_comm);
        
        // Print error every 100 iterations
        if (rank == 0 && iteration % 100 == 0) {
            printf("Iteration %d global_error = %g\n", iteration, global_error);
        }
        
        // Check convergence
        if (global_error < tolerance) {
            if (rank == 0) {
                double end_time = MPI_Wtime();
                printf("Converged after %d iterations in %f seconds\n", 
                       iteration + 1, end_time - start_time);
            }
            break;
        }
        
        // Swap pointers
        double *temp = u;
        u = u_new;
        u_new = temp;
    }
    
    // Output results for process 0
    if (rank == 0) {
        output_results(u, u_exact);
    }
    
    // Cleanup
    free(u);
    free(u_new);
    free(u_exact);
    // Note: f is allocated in initialization, so it should be freed somewhere
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}
