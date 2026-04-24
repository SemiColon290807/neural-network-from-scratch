# neural-network-from-scratch

## What This Is
My first attempt at building a neural network from scratch.

## Phase 1: Matrix Operations
### Functions Implemented
- scalar_multiply(A, c): multiplies every element A_ij by scalar c
- transpose(A): maps element A_ij → A_ji, converts m×n to n×m matrix
- dot_product(u, v): computes Σ u_i * v_i, projects one vector onto another
- matrix_multiply(A, B): element (i,j) = dot product of row i of A with col j of B

## Benchmark Results
| Method     | Time (500×500) |
|------------|----------------|
| Raw Python | 4.608s         |
| NumPy      | 0.00188s       |
| Speedup    | ~2450x         |

## Why NumPy is Faster
In the matrix_numpy.py file, I implemented the basic matrix operations using the functions in the numpy library which has all the operations already implemented using compiled C code and not python. This makes the execution of a 500x500 matrix multiplication much faster -- in 0.0018801689147949219 seconds.
