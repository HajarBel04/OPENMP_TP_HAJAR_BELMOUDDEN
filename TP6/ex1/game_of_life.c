#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define ALIVE 1
#define DEAD 0

// Function to initialize the grid with a random pattern
void initialize_grid(int *grid, int sx, int ex, int sy, int ey, int rank) {
    srand(rank + 1);  // Seed with rank for reproducibility
    
    int local_nx = ex - sx + 3;  // +3 for halos on both sides
    int local_ny = ey - sy + 3;
    
    // Initialize all to zero first
    for (int i = 0; i < local_nx * local_ny; i++) {
        grid[i] = 0;
    }
    
    // Initialize only the internal cells (not halo)
    for (int i = 1; i <= ex - sx + 1; i++) {
        for (int j = 1; j <= ey - sy + 1; j++) {
            grid[i * local_ny + j] = (rand() % 100 < 30) ? ALIVE : DEAD;
        }
    }
}

// Function to count neighbors
int count_neighbors(int *grid, int i, int j, int local_ny) {
    int count = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            count += grid[(i + di) * local_ny + (j + dj)];
        }
    }
    return count;
}

// Function to apply Game of Life rules
void apply_game_of_life_rules(int *old_grid, int *new_grid, int local_nx, int local_ny) {
    for (int i = 1; i <= local_nx - 2; i++) {
        for (int j = 1; j <= local_ny - 2; j++) {
            int neighbors = count_neighbors(old_grid, i, j, local_ny);
            int cell = old_grid[i * local_ny + j];
            
            // Apply Conway's rules
            if (cell == ALIVE) {
                if (neighbors < 2 || neighbors > 3) {
                    new_grid[i * local_ny + j] = DEAD;
                } else {
                    new_grid[i * local_ny + j] = ALIVE;
                }
            } else {
                if (neighbors == 3) {
                    new_grid[i * local_ny + j] = ALIVE;
                } else {
                    new_grid[i * local_ny + j] = DEAD;
                }
            }
        }
    }
}

// Function to print local grid
void print_local_grid(int *grid, int local_nx, int local_ny, int rank, int generation) {
    printf("Rank %d - Generation %d:\n", rank, generation);
    for (int i = 1; i <= local_nx - 2; i++) {
        for (int j = 1; j <= local_ny - 2; j++) {
            printf("%d ", grid[i * local_ny + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int rank, size;
    int dims[2] = {0, 0};
    int periods[2] = {1, 1};  // Periodic boundaries
    int reorder = 0;
    int coords[2];
    int north, south, east, west;
    
    // Global grid dimensions
    int ntx = 24;  // Global grid size in x
    int nty = 24;  // Global grid size in y
    int generations = 10;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create 2D Cartesian communicator
    MPI_Dims_create(size, 2, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get coordinates in Cartesian communicator
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Calculate local grid boundaries using professor's approach
    int sx = (coords[0] * ntx) / dims[0] + 1;
    int ex = ((coords[0] + 1) * ntx) / dims[0];
    int sy = (coords[1] * nty) / dims[1] + 1;
    int ey = ((coords[1] + 1) * nty) / dims[1];
    
    int local_nx = ex - sx + 3;  // +3 for halos
    int local_ny = ey - sy + 3;
    
    if (rank == 0) {
        printf("Creating %d x %d process grid for %d processes\n", dims[0], dims[1], size);
    }
    
    printf("Rank %d: Coordinates (%d, %d), Grid indices: %d to %d in x, %d to %d in y\n", 
           rank, coords[0], coords[1], sx, ex, sy, ey);
    
    // Find neighbors
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    
    // Allocate grids
    int *grid = (int *)calloc(local_nx * local_ny, sizeof(int));
    int *new_grid = (int *)calloc(local_nx * local_ny, sizeof(int));
    
    // Initialize grid
    initialize_grid(grid, sx, ex, sy, ey, rank);
    
    // Create MPI datatype for column exchange
    MPI_Datatype column_type;
    MPI_Type_vector(local_nx - 2, 1, local_ny, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);
    
    // Main simulation loop
    for (int gen = 0; gen < generations; gen++) {
        // Exchange halos with neighbors
        
        // North-South exchange
        MPI_Sendrecv(&grid[1 * local_ny + 1], local_ny - 2, MPI_INT, north, 0,
                     &grid[(local_nx - 1) * local_ny + 1], local_ny - 2, MPI_INT, south, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&grid[(local_nx - 2) * local_ny + 1], local_ny - 2, MPI_INT, south, 1,
                     &grid[0 * local_ny + 1], local_ny - 2, MPI_INT, north, 1,
                     cart_comm, MPI_STATUS_IGNORE);
        
        // West-East exchange
        MPI_Sendrecv(&grid[1 * local_ny + 1], 1, column_type, west, 2,
                     &grid[1 * local_ny + (local_ny - 1)], 1, column_type, east, 2,
                     cart_comm, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&grid[1 * local_ny + (local_ny - 2)], 1, column_type, east, 3,
                     &grid[1 * local_ny + 0], 1, column_type, west, 3,
                     cart_comm, MPI_STATUS_IGNORE);
        
        // Apply Game of Life rules
        apply_game_of_life_rules(grid, new_grid, local_nx, local_ny);
        
        // Swap grids
        int *temp = grid;
        grid = new_grid;
        new_grid = temp;
        
        // Print local grid for final generation
        if (gen == generations - 1) {
            print_local_grid(grid, local_nx, local_ny, rank, gen);
        }
    }
    
    // Gather final grid to process 0 (optional)
    if (rank == 0) {
        int *global_grid = (int *)malloc(ntx * nty * sizeof(int));
        
        // Copy process 0's data
        for (int i = 1; i <= local_nx - 2; i++) {
            for (int j = 1; j <= local_ny - 2; j++) {
                int global_i = sx + i - 2;
                int global_j = sy + j - 2;
                global_grid[global_i * nty + global_j] = grid[i * local_ny + j];
            }
        }
        
        // Receive from other processes
        for (int p = 1; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            
            int p_sx = (p_coords[0] * ntx) / dims[0] + 1;
            int p_ex = ((p_coords[0] + 1) * ntx) / dims[0];
            int p_sy = (p_coords[1] * nty) / dims[1] + 1;
            int p_ey = ((p_coords[1] + 1) * nty) / dims[1];
            
            int p_local_nx = p_ex - p_sx + 1;
            int p_local_ny = p_ey - p_sy + 1;
            
            int *recv_buffer = (int *)malloc(p_local_nx * p_local_ny * sizeof(int));
            MPI_Recv(recv_buffer, p_local_nx * p_local_ny, MPI_INT, p, 0, cart_comm, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < p_local_nx; i++) {
                for (int j = 0; j < p_local_ny; j++) {
                    int global_i = p_sx + i - 1;
                    int global_j = p_sy + j - 1;
                    global_grid[global_i * nty + global_j] = recv_buffer[i * p_local_ny + j];
                }
            }
            
            free(recv_buffer);
        }
        
        printf("\nGlobal Grid after %d generations:\n", generations);
        for (int i = 0; i < ntx; i++) {
            for (int j = 0; j < nty; j++) {
                printf("%d ", global_grid[i * nty + j]);
            }
            printf("\n");
        }
        
        free(global_grid);
    } else {
        // Send local data to process 0
        int *send_buffer = (int *)malloc((local_nx - 2) * (local_ny - 2) * sizeof(int));
        for (int i = 1; i <= local_nx - 2; i++) {
            for (int j = 1; j <= local_ny - 2; j++) {
                send_buffer[(i - 1) * (local_ny - 2) + (j - 1)] = grid[i * local_ny + j];
            }
        }
        MPI_Send(send_buffer, (local_nx - 2) * (local_ny - 2), MPI_INT, 0, 0, cart_comm);
        free(send_buffer);
    }
    
    // Cleanup
    MPI_Type_free(&column_type);
    free(grid);
    free(new_grid);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}
