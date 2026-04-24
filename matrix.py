import random
import time
def scalar_multiply(A,c):
    result=[]
    for row in A:
        new_row= []
        for element in row:
            new_row.append(element*c)
        result.append(new_row)
    return result

def transpose(A):
    rows= len(A[0])
    cols=len(A)
    result= [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i]=A[i][j]
    return result
        
def dot_product(u,v):
    if len(u)!=len(v):
        raise ValueError("Vectors must be of compatible length!")
    result=0
    for i in range(len(u)):
        result+=u[i]*v[i]
    return result

def matrix_multiply(a,b):
    if len(a[0])!=len(b):
        raise ValueError("Matrices are not compatible!")
    result=[]
    for row_A in a:

        new_row= []
        for j in range(len(b[0])):
            col=[]
            for row in b:
                col.append(row[j])
            val= dot_product(row_A,col)
            new_row.append(val)
        result.append(new_row)
    return result

if __name__=="__main__":
    size=500
    A= [[random.random() for _ in range(size)] for _ in range(size)]
    B= [[random.random() for _ in range(size)] for _ in range(size)]
    start_time= time.time()
    matrix_multiply(A,B)
    end_time= time.time()

    print(f"Time taken: {end_time-start_time} seconds")

