# Shallow Neural Network

[TOC]

### Neural Network Overview

#### What is a neural network?

as we saw in th last week in logistic regression model  which is built out of single sigmoid unit as where you input the the feature x as well as the parameters  w and b compute the linear combination Z and the apply the sigmoid function to Z and ultimately find loss between the prediction and the actual value as shown in figure (1):

![Screenshot from 2020-07-17 07-26-05](/home/sa3eed/Pictures/Screenshot from 2020-07-17 07-26-05.png)

​																										figure (1): neural network overview



on the other side, neural network is a stack set of logistic regression units as show figure 1 but there are alternative choices for activation functions to be used other than sigmoid activation.

each unit in the network is responsible for performing two steps of calculations in a single layer which are (Z and a) calculations and the output of the layer act as an input to the next layer until we reach the ultimate unit and compute the loss, such issue requires introducing new notation to indicate the layer, so that we will use superscript squared brackets to indicate certain layer and its calculation such as W^[1],  b^[1], z^[1] and a^[1]  as shown in figure 1. each layer has its own parameters (W and b) and thus, back propagation is then performed with respect to each parameter in the network from the last layer to the first layer.



### Neural Network Representation?

#### what do the part of neural network mean?

![Screenshot from 2020-07-17 08-21-43](/home/sa3eed/Pictures/Screenshot from 2020-07-17 08-21-43.png)

​																											figure (2): neural network representation

figure 2 represents a simple neural network, the first part of the layer is called the **input layer** which hold the input feature values without any activation functions, the second layer is called the **hidden layer** which is responsible for  linearly combining the input variables and then apply the activation to it, and the last one is called the **output layer** which is responsible for predicting the output value. to have better idea about the hidden layer and what does it mean, as you that training set contain both input features values (input layers) as well as the class label (output layer) which are explicitly known for you but in case of hidden layer its input and output is not readily available for you in the training set and the are inferred during the training phase, for this reason it is called a hidden layer.



lets introduce more notation,  in logistic regression we were using vector X as input features representative, alternatively, the input feature vector will be represented by a^[0] , the hidden layer will be denoted by a^[1] which will generate the values a^[1]_1,  a^[1]_2, a^[1]3, a^[1]4  and finally, the output layer is called a^[2].

such structure in figure 2 is called **(2 layers neural network)** despite that we have defined 3 layers input hidden and output. the reason behind that is we do not count the input layer as an official layer because it does not have any activation function. 

on more thing to mention is that the hidden and the output layers will have parameters (W and b) associated with each of them and the will be indicated as according to the notation of certain layer. for example the hidden layer a^[1] will have the parameters w^[1] and b^[1] and so on. later on we will talk in details about the dimensions of each vector.



### Computing a Neural Network's Output

#### 