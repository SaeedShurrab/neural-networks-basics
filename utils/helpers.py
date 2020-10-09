import numpy as np
import os

# Activation functions defination
def activation_fun(x,activation_type= 'sigmoid'):
    
    if activation_type not in ['sigmoid','tanh', 'relu','lrelu']:
        raise ValueError(" activation type must be in ['sigmoid','tanh', 'relu','lrelu']")
    
    if activation_type == 'sigmoid':
        return (1/(1+np.exp(-x)))
    
    elif activation_type == 'tanh':
        return np.tanh(x)
    
    elif activation_type == 'relu':
        return np.maximum(0.0,x)
    
    elif activation_type == 'lrelu':
        return np.maximum(0.01 * x, x)

def act_derivative(x, activation_type= 'sigmoid'):

    if activation_type not in ['sigmoid','tanh', 'relu','lrelu']:
        raise ValueError(" activation type must be in ['sigmoid','tanh', 'relu','lrelu']")    
    
    if activation_type == 'sigmoid':
        return activation_fun(x, activation_type= activation_type)\
        * (1- activation_fun(x, activation_type=activation_type))
    
    elif activation_type == 'tanh':
        return 1- (activation_fun(x,activation_type= activation_type))**2
    
    elif activation_type == 'relu':
        return np.where(x <= 0.0, 0.0,1.0)
    
    elif activation_type == 'lrelu':
        return np.where(x <= 0.0, 0.01 * x,1.0)


# Random data generation for function
def data_generator(num_features=4, num_examples=1000,train_p=0.8):
    
    dataset = np.random.randn(num_features,num_examples)
    labels = np.random.randint(0,2,(1,num_examples))
    
    x_train = dataset[:,:int(num_examples * train_p)]
    y_train = labels[:,:int(num_examples * train_p)]
    x_test = dataset[:,int(num_examples * train_p) :]
    y_test = labels[:,int(num_examples * train_p):]
    return x_train, y_train, x_test, y_test