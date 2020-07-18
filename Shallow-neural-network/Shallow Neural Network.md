# Shallow Neural Network

[TOC]

### Neural Network Overview

#### What is a neural network?

as we saw in th last week in logistic regression model  which is built out of single sigmoid unit as where you input the the feature x as well as the parameters  w and b compute the linear combination Z and the apply the sigmoid function to Z and ultimately find loss between the prediction and the actual value as shown in figure (1):

![Screenshot from 2020-07-17 07-26-05](/home/sa3eed/Pictures/Screenshot from 2020-07-17 07-26-05.png)

​																										figure (1): neural network overview



on the other side, neural network is a stack set of logistic regression units as show figure 1 but there are alternative choices for activation functions to be used other than sigmoid activation.

each unit in the network is responsible for performing two steps of calculations in a single layer which are (Z and a) calculations and the output of the layer act as an input to the next layer until we reach the ultimate unit and compute the loss, such issue requires introducing new notation to indicate the layer, so that we will use squared brackets to indicate certain layer and its calculation such as W[1],  b[1], z[1] and a[1]  as shown in figure 1. each layer has its own parameters (W and b) and thus, back propagation is then performed with respect to each parameter in the network from the last layer to the first layer.



### Neural Network Representation?

#### what do the part of neural network mean?

![Screenshot from 2020-07-17 08-21-43](/home/sa3eed/Pictures/Screenshot from 2020-07-17 08-21-43.png)

​																											figure (2): neural network representation

figure 2 represents a simple neural network, the first part of the layer is called the **input layer** which hold the input feature values without any activation functions, the second layer is called the **hidden layer** which is responsible for  linearly combining the input variables and then apply the activation to it, and the last one is called the **output layer** which is responsible for predicting the output value. to have better idea about the hidden layer and what does it mean, as you that training set contain both input features values (input layers) as well as the class label (output layer) which are explicitly known for you but in case of hidden layer its input and output is not readily available for you in the training set and the are inferred during the training phase, for this reason it is called a hidden layer.



lets introduce more notation,  in logistic regression we were using vector X as input features representative, alternatively, the input feature vector will be represented by a[0] , the hidden layer will be denoted by a[1] which will generate the values a[1]_1,  a[1]_2, a[1]3, a[1]4  and finally, the output layer is called a[2].

such structure in figure 2 is called **(2 layers neural network)** despite that we have defined 3 layers input hidden and output. the reason behind that is we do not count the input layer as an official layer because it does not have any activation function. 

on more thing to mention is that the hidden and the output layers will have parameters (W and b) associated with each of them and the will be indicated as according to the notation of certain layer. for example the hidden layer a[1] will have the parameters w[1] and b[1] and so on. later on we will talk in details about the dimensions of each vector.



### Computing a Neural Network's Output

#### The neural network basic computation

![Screenshot from 2020-07-18 10-23-14](/home/sa3eed/Pictures/Screenshot from 2020-07-18 10-23-14.png)

​																				figure (3): What is going on inside the neural network??

In Logistic regression (LHS of figure 3), each single unit performs two steps of calculations including:

```
z =w^T * x +b			(1)
a = sigmoid(z)			(2)
```

The same computation is performed in the neural network but multiple times according to the number of units i the neural network, consider the RHS of figure (3) which represents a shallow neural network with three input variables, single training example, single hidden layer with four units and output layer with single unit, lets dive deeper into the calculation of each unit in the network, we well take the first unit as an example.

Similar to logistic regression,  equation  (1) and (2) are performed in the each unit in the network. bear in mind that the number in the squared brackets represents the layer index while the subscripted number represents the index of the unit in that layer.

```
for the first unit:
z[1]_1 = w^T[1]_1 * x + b[1]_1			(1)
a[1]_1 = sgmoid(z[1]_1)					(2)

for the second unit:
z[1]_2 = w^T[1]_2 * x + b[1]_2			(3)
a[1]_2 = sgmoid(z[1]_2)					(4)

for the third unit:
z[1]_3 = w^T[1]_3 * x + b[1]_3			(5)
a[1]_3 = sgmoid(z[1]_3)					(6)

for the fourth unit:
z[1]_4 = w^T[1]_4 * x + b[1]_4			(7)
a[1]_4 = sgmoid(z[1]_4)					(8)
```

to perform these computations, you need start a for loop to do all the computation (1-8), but using for loop is not efficient practice with neural networks, so lets vectorize these equations:

```
for the hidden layer:
vectorizing the linear equations(1, 3, 5, 7)
w^T[i]_j is a row vector of shape (1,n) or (1,3) in our example, in vectorized version the weight vectors are stacked
together to form a matrix of shape (r[i],n) or (4,3) in our example where (r[i]) represents the number of units in the
layers and (n) is the number of input variables, then the weights matrix is multiplied by the input matrix which has
the shape (n,1) or (3,1) in our case, ultimately, the bias vector of shape (r[1],1) to produce a column vector contains
the Z's values with shape of (r[1],1) or (4,1) in our case. 
	(r[1],1)			  (r[1],n)		(n,1)		  (r[1],1)	
    (4,1)				  (4,3)			(3,1)		  (4,1)
	Z[1]			=	  W[1]		*	  x 		  b[1]
z[1] = 	[z[1]_1, 	=	[w^T[1]_1,	*	[x_1,	+	[b[1]_1,		= 	[w^T[1]_1 * x + b[1]_1,		
 	 	 z[1]_2,		 w^T[1]_2,	 	 x_2,		 b[1]_2,			 w^T[1]_2 * x + b[1]_2,
 	 	 z[1]_3,		 w^T[1]_3,	 	 x_3]		 b[1]_3,			 w^T[1]_3 * x + b[1]_3,
 	 	 z[1]_4]	 	 w^T[1]_4]					 b[1]_4]			 w^T[1]_4 * x + b[1]_4]

vectorizing the linear equations(2, 4, 6, 8)
applying the sigmoid function element-wisely to z[1] will produce a column vector of shape(r,1)
		(r[1],1)
		(4,1)
a[1] = [a[1]_1,
		a[1]_2,
		a[1]_3,
		a[1]_4]
		
for the output layer:
the output layer has the parameter w[2] of shape (1,4) and b of shape (1,1)
(r[2],1)   (r[2],r[1])  (r[1],1) + (r[2],1)
(1,1)  		(1,4)  		(4,1) 	 + (1,1)
z[2] = 		W[2] 	* 	 a[1]    +  b[2]

(r[2],1)    (r[2],1)
(1,1)		(1,1)
a[2]	= sigmoid(z[2])
```

remember from the previous section that we indicated x as a[0] vector so that, this notation can substitute x in our equations



### Vectorizing Along Multiple examples

#### Completely vectorized shallow neural network

in the previous section we saw how to find a[2] = y-hat for a single training example, in this section we will complete the same mission but for (m) training examples, let's see how:

```
to find predict the prediction of all training example we have to do the following computations:
x_1 ===============> a[2]_1 = y-hat
x_2 ===============> a[2]_2 = y-hat
x_3 ===============> a[2]_3 = y-hat
x_4 ===============> a[2]_4 = y-hat
.
.
x_m ===============> a[2]_m = y-hat

which can be obtained by for loop through each example:
for i 1 to m:
	z[1]_i = w[1] * x_i +b[1]
	a[1]_i = sigmod(z[1]_i)
	z[2]_i = w[2] * a[1]_i +b[2]
	a[2]_i = sigmod(z[2]_i)
	
Vectorized implementation:
recall that X is a matrix of training examples stacked as columns with shape of (n,m) as follow:
X = [x_1, x_2, ......x_m ]
to vectorize the above for loop we have to compute:

Z[1] = W[1] * X +b[1]
A[1] = sigmod(Z[1])
Z[2] = W[2] * A[1] +b[2]
A[2] = sigmod(Z[2])

substituting single training example by m training example will result in 
matrix Z[1] and A[1] of shape (r[1],m) where each column represents the computations of a 
single training example. in addition Z[2] and A[2] are matrix of shape (r[2],m)

```



### Explanation for Vectorized Implementation

#### Vectorized Implementation justified

```
for each training example you end up computing :
z[1]_1 = w^T[1] * x_1 + b[1]
z[1]_2 = w^T[1] * x_2 + b[1]
z[1]_m = w^T[1] * x_3 + b[1]

w
W[1] is a matrix of shape (r[i],r[j])
and perform 
w^T[1] * x_1 = column vector
w^T[1] * x_2 = column vector
w^T[1] * x_m = column vector 

X is a matrix of all training examples stacked as column for each example and hence
W[1] * X  = will results in a matrix that contains the Z values stacked as columnns

= [z[1]_1 z[1]_2 z[1]_m] = Z[1]
```

