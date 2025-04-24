#ifndef COMPUTE_H
#define COMPUTE_H

// Global variables
extern int sx, ex, sy, ey, ntx, nty;

// Function declarations
void initialization(double **pu, double **pu_new, double **pu_exact);
void compute(double *u, double *u_new);
void output_results(double *u, double *u_exact);

#endif
