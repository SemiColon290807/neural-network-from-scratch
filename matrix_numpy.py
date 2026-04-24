import numpy as np
import random
import time
def scalar_multiply(A,c):
    return c*A
def transpose(A):
    return np.transpose(A)
def dot_product(u,v):
    return np.dot(u,v)
def matrix_multiply(a,b):
    return a@b

if __name__=="__main__":

    size= 500
    A= np.random.rand(size,size)
    B= np.random.rand(size,size)
    start_time=time.time()
    matrix_multiply(A,B)
    end_time=time.time()
    print(f"Time taken: {end_time-start_time} seconds")