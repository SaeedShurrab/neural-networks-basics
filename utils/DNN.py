import numpy as np
import matplotlib.pyplot as plt
from helpers import activation_fun 
from helpers import act_derivative
from helpers import data_generator



class DeepNeuralNet:
    
    xtr, ytr, xts, yts = data_generator()

    def __init__(self, X=xtr,Y=ytr, activation_order= ['relu','relu','sigmoid'], layer_dims = [4,3,2,1], num_iterations = 1000, learning_rate =.005, print_cost = False):
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims
        self.activation_order = activation_order
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
    

# parameters initiation
    def initialize_parameters(self,layer_dims):
    
        parameters ={}
        L = len(layer_dims)
        for l in range(1,L):
            parameters['W'+ str(l)] =  np.random.randn(layer_dims[l],layer_dims[l-1])*0.1
            parameters['b'+ str(l)] =  np.zeros((layer_dims[l],1))
    
        return parameters


    def forward_prop(self,A_prev, parameters, activation_order):
    
        cache = {}
    
        for  l,activation in enumerate(activation_order):
        
            W = parameters['W' + str(l+1)]
            b = parameters['b' + str(l+1)]
       
            Z = np.dot(W,A_prev) + b
            A = activation_fun(Z,activation_type= activation)
        
            cache['Z' + str(l+1)]= Z
            cache['A' + str(l+1)]= A
        
            A_prev = A
    
        return cache


    def compute_cost(self,Y, cache):
    
        AL = cache[(list(cache.keys())[-1])]
        m = Y.shape[1]
    
        cost = np.squeeze((-1/m)*(np.dot(Y,np.log(AL).T)+ np.dot((1-Y),np.log(1-AL).T)))
        assert(cost.shape == ())
    
        return cost

    def backward_prop(self,X,Y,parameters,cache, activation_order):
    
        AL = cache['A'+str(len(activation_order))]
        grads = {}
        dA =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        m = Y.shape[1]
    
    
        for l, activation in reversed(list(enumerate(activation_order))):
        
            Z = cache['Z' + str(l+1)]
        
            if l == 0:
                A_prev = X
            else:
                A_prev = cache['A' + str(l)]
            
            W = parameters['W' + str(l+1)]
        
            dZ = dA * act_derivative(Z, activation_type= activation)
            dW = (1/m) * np.dot(dZ,A_prev.T)
            db =  (1/m) * np.sum(np.dot(dZ, A_prev.T), axis = 1, keepdims =True)
            dA = np.dot(W.T,dZ)
        
            grads['dW'+ str(l+1)] = dW
            grads['db'+ str(l+1)] = db
    
        return grads

    def optimize(self,parameters, grads, learning_rate = 0.01):
    
        L = int(len(grads)/2)+1
    
        for l in range(1,L):
        
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
        
            dW = grads['dW' + str(l)]
            db = grads['db' + str(l)]
        
            parameters['W' + str(l)] = W - learning_rate * dW
            parameters['b' + str(l)] = b - learning_rate * db
    
        return parameters


    def fit(self):
        global parameters
        parameters = self.initialize_parameters(self.layer_dims)
    
        for i in range(self.num_iterations):
    
            cache = self.forward_prop(self.X, parameters, self.activation_order)
        
            cost = self.compute_cost(self.Y,cache)
        
            grads = self.backward_prop(self.X,self.Y,parameters,cache,self.activation_order)
        
            parameters = self.optimize(parameters,grads,self.learning_rate)
        
            if self.print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
    
        

    def predict(self, X=xts,Y=yts):
    
        m = X.shape[1]
        p = np.zeros((1,m))
    
        # Forward propagation
        cache = self.forward_prop(X,parameters,self.activation_order)
        probas = cache[(list(cache.keys())[-1])]

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
    
        print("Accuracy: "  + str(np.sum((p == Y)/m)))    
