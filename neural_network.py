import numpy as np
def sigmoid(z):
    s= 1/(1+np.exp(-z))
    return s

def relu(z):
    return np.maximum(0,z)

def sigmoid_derivative(z):
    sd= sigmoid(z)*(1-sigmoid(z))
    return sd

def relu_derivative(z):
    return 1 if (z>0) else 0
     
class Layer:

    def __init__(self,input_size,output_size):
        rows= input_size
        cols= output_size
        self.weights= np.random.randn(rows,cols)*np.sqrt(2.0/rows)
        self.biases= np.zeros(cols)
    
    def forward(self,inputs):
        self.inputs=inputs
        self.z= self.inputs@self.weights+self.biases
        return relu(self.z)
    
    def backward(self, dL_dy, l_rate): #dL_dy represents the derivative from the previous layer propagated backwards.
        dL_dz= dL_dy*(self.z>0)
        dL_dW= self.inputs.T@ dL_dz
        dL_db= np.sum(dL_dz, axis=0)
        dL_dX= dL_dz@ self.weights.T
        self.weights-= l_rate*dL_dW
        self.biases-= l_rate*dL_db
        return dL_dX
    
class NeuralNetwork:
    def __init__(self,layer_sizes):
        self.layers= []
        for i in range(len(layer_sizes)-1):
            new_layer= Layer(layer_sizes[i],layer_sizes[i+1])
            self.layers.append(new_layer)

    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
            
        return x
    
    def backward(self,dL_dy, l_rate):
        for layer in reversed(self.layers):
            dL_dy= layer.backward(dL_dy, l_rate)
            
