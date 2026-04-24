# neural-network-from-scratch
My first attempt at building a neural network from scratch.

In the matrix.py file, I have implemented basic matrix operations(scalar multiplication, transposition, dot product and matrix multiplication) from scratch in raw python. The time taken to perform a 500x500 matrix multiplication in raw python took 4.608090162277222 seconds, a surprisingly long time.
On the other hand in the matrix_numpy.py file I have done the same using the numpy library which has all the operations already implemented using compiled C code and not python. This makes the execution of a 500x500 matrix multiplication much faster -- in 0.0018801689147949219 seconds, much faster than before.
