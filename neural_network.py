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

def softmax(z):
    shifted= z- np.max(z, axis=1, keepdims= True)
    exp_z= np.exp(shifted)
    return exp_z/ np.sum(exp_z, axis=1, keepdims= True)
    
def cross_entropy_loss(y_pred, y_true):
    clip= np.clip(y_pred, 1e-15, 1-1e-15)
    return -np.mean(np.sum(y_true*np.log(clip),axis=1))

def cross_entropy_loss_derivative(y_pred, y_true):
    return (y_pred-y_true)/y_true.shape[0]

class Layer:

    def __init__(self,input_size,output_size,activation_type):
        rows= input_size
        cols= output_size
        self.activation_type= activation_type
        if self.activation_type== 'relu':
            self.weights= np.random.randn(rows,cols)*np.sqrt(2.0/rows)
        else:
            self.weights= np.random.randn(rows,cols)*np.sqrt(1.0/rows)
        self.biases= np.zeros(cols)
        
    
    def forward(self,inputs):
        self.inputs=inputs
        self.z= self.inputs@self.weights+self.biases
        if self.activation_type=='relu':
            return relu(self.z)
        if self.activation_type== 'softmax':
            return softmax(self.z)
    
    def backward(self, dL_dy, l_rate): #dL_dy represents the derivative from the previous layer propagated backwards.
        if self.activation_type== 'relu':
            dL_dz= dL_dy*(self.z>0)
        if self.activation_type== 'softmax':
            dL_dz= dL_dy
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
            if i <(len(layer_sizes)-2):
                activation= 'relu'
            else:
                activation= 'softmax'
            new_layer= Layer(layer_sizes[i],layer_sizes[i+1], activation)
            self.layers.append(new_layer)
        
        

    def forward(self,x):
        for layer in self.layers:
            x=layer.forward(x)
            
        return x
    
    def backward(self,dL_dy, l_rate):
        for layer in reversed(self.layers):
            dL_dy= layer.backward(dL_dy, l_rate)
            
    
