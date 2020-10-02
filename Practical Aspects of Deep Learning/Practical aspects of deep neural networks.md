# Practical aspects of deep neural networks

[TOC]

# Setting up your deep learning application

### Train, Validation and Test set 

#### Deep learning is a highly iterative process

when training a NN you have to set a bunch of specifications (hyper-parameters) for your model to get satisfactory results such as:

1. number of layers
2. number of hidden layers
3. learning rate
4. choice of activation function

so that it is impossible to correctly select them from the first experiment, so that you have configure your sittings iteratively and observe the change in your model performance, for this reason Deep learning is a highly iterative process.



#### Train/ val/ test

if you have a certain dataset from certain domain, you may want yo divide it as follow:

1. training data
2. validation data
3. test data

having this data division, then you can train your model on the **training ** data, examine which of your models perform better on the **validation** data and ultimately evaluate your final model on the **test** data in order to get unbiased estimates of how well your model is doing.

previously, it was a common practice to divide your data as 70/15/15 train-val-test splits or 60/20/20 splits if you have small amount of data 100-10000. such procedure form a best practice to train your model.

Currently in the era of big data (millions of data points), the trend is oriented toward having smaller percentages of validation and test sets. remember that the goal of validation set is to test different model setting on it and see which model works better while the main goal of your test set is to have pretty confident estimates of how well it is doing. and thus if you have 1 million examples you do not have to have  a very large validation and test sets, (1-2) % splits percentages are fair enough.



#### Mismatched train/test distribution

assume we have a model that predict whether an input image is a cat or not cat. Further, your training data is of high resolution while your test and validation data sets are of low resolution. and thus there will have a different distribution between your training data and validation test data. To overcome this issue, make sure that all data splits are coming from the same distribution



One more thing to note is :

it OK to have only **validation set ** rather than **validation and test sets**, remember that the goal of your test set is to get unbiased estimates of your model performance. if you do not need that unbiased estimate then it might be OK to omit the test set and use validation set rather. In this case you have to train on the training set and evaluate on the validation set.



### Bias / Variance

#### Bias and variance explained

![](../assets/4fig1.png)

​																										Figure(1) : Bias Vs Variance 

 if we have a dataset that looks like in figure (1). if you fit a straight line to the data we can see that the best fitting line is biased toward the red labels causing the model to be **Underfitted**. On the other side, if you fit an incredibly complex model, may be you can fit the data perfectly but that does not look a great fit either  as there is a high variance in the model which cause the model to be **Overfitted**. In between, having a model with medium level of complexity (bias variance traded-off)  will give a reasonable estimates. 

However this is a simple 2D problem where we are able to visualize the data and observe the model. In higher dimensional this is not achievable. So that, we will look at two other metrics which are :

1. Train set error
2. Validation set error
3. Bayes error

these two measures can be compared simultaneously to observe the model performance in term bias and variance. lets see in details how these measures works

| Train set error | Validation set error |      Model performance      |
| :-------------: | :------------------: | :-------------------------: |
|       1%        |         11%          | High variance (overfitting) |
|       15%       |         16%          |  High bias (Under fitting)  |
|       15%       |         30%          |  High bias - high variance  |
|      0.5%       |          1%          |   Low bias - low variance   |



- Case (1): In this case, the model performs very well on the training set but poorly on the validation set causing the model to overfit the training data but fails to generalized on the validation set. Ultimately you can conclude that your model has high variance.
- Case (2): in this case, the model preforms badly on the training set causing the model to be Underfitted and so the model has high bias because it was not even fitting the training set. In contrast it generalize well on the validation set as the validation set error is worse than training error by only 1% .
- Case (3): In this case. the model performs poorly on both training and validation sets. And thus it has a high bias be cause it performs poorly on the training set and high variance as it performs poorly on the test set.
- Case (4): In this case, the model performs very well and both datasets. And thus it has low bias and low variance. In fact, this kind of performance we wish to achieve.



**Note**

Pear in mind that such kind of analysis is performed with respect to the **Human Level performance** on the same dataset which called **Bayes error** which is hypothetically equal to zero (0) and vary from domain to domain. For example, if our Bayes error is 15%   then we can consider the **second** model as the optimal performance in the previously showed 4 cases.



To sum up, examining the training set error gives a sense bias problem while examining the validation set error gives a sense of variance problem.



### Basic recipe for deep learning

#### Recipe elements

the basic recipe provides the practitioner with a systematic approach to improve the model performance and reduce both variance and bias as follow:



**High Bias?**

Look at the training data performance and try single or a combination of these solutions to improve the model performance on the training data:

1. Try a bigger network (number of layers and number of units).

1. Train the model for longer time.
2. Use different architectures (RNN, CNN , etc).



**High Variance?**

Look at the validation data performance and try single or a combination of these solutions to improve the model performance on the training data:

1. Add more data.
2. Employ regularization techniques.
3. Use different architectures (RNN, CNN , etc).



A couple of points to notice:

First is that, depending on whether you have high **bias** or high **variance**, the set of solutions you should try could be quite different. So I'll usually use the training/ validation sets to try to diagnose if I have a bias or variance problem, and then use that to select the appropriate subset of solutions to try. So for example, if you actually have a high bias problem, getting more training data is actually not going to help. Or at least it's not the most efficient thing to do. So being clear on how much of a bias problem or variance problem or both can help you focus on selecting the most useful solution to try. 

Second, in the earlier era of machine learning, there used to be a lot of discussion on what is called the bias variance **tradeoff**. And the reason for that was that, for a lot of the solution you could try, you could increase bias and reduce variance, or reduce bias and increase variance. But back in the pre-deep learning era,  we didn't have as many tools that just reduce bias or that just reduce variance without hurting the other one. But in the modern deep learning, big data era, so long as you can keep training a bigger network, and so long as you can keep getting more data, which isn't always the case for either of these, but if that's the case, then getting a bigger network almost always just reduces your bias without necessarily hurting your variance, so long as you regularize appropriately. And getting more data pretty much always reduces your variance and doesn't hurt your bias much. 

So what's really happened is that, with these two steps, the ability to train, pick a network, or get more data, we now have tools to drive down bias and just drive down bias, or drive down variance and just drive down variance, without really hurting the other thing that much. And I think this has been one of the big reasons that deep learning has been so useful for supervised learning, that there's much less of this tradeoff where you have to carefully balance bias and variance, but sometimes you just have more options for reducing bias or reducing variance without necessarily increasing the other one. 



# Regularizing your neural network

### Regularization 