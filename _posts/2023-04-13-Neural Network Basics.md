---
layout: post
title: Neural Networks Basics
date: 2023-04-13
categories: Neural Networks
tags: [Neural Networks, Deep Learning]
---

![header image](/assets/img/post_imgs/neural_network_basics/header.jpg)

## Intoduciton:

<p> A neural network is a computer program that tries to learn from data, similar to how humans learn from experience. It is made up of a series of interconnected nodes, or neurons, that work together to solve a specific task. Each neuron receives inputs from other neurons or directly from the data, and uses a set of weights and biases to process the information and produce an output.</p>

<p>Once trained, the network can be used to make predictions or decisions based on new input data.In simple terms a neural network is basically a set of functions which can learn to recognize and generalize patterns in data,making them a powerful tool for a wide range of machine learning applications from image recognition and speech recognition to autonomous vehicles and natural language processing.</p>

<p>Let's Create a simple neural network with single neuron with no additional layers to understand it's working</p>

<p>we are going to write a code using Python and TensorFlow and an API in TensorFlow called keras. Keras makes it really easy to define neural networks.</p>

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
np.random.seed(42)

import datetime, os
import matplotlib.pyplot as plt

print(tf.__version__)

```
### Note : here i am using simple network midel from Deepleaning.ai 

## Creating the model

<p>The simplest possible neural network is one that has only one neuron in it, and that's what this line of code does.

In keras, you use the word dense to define a layer of connected neurons. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron.

Successive layers are defined in sequence, hence the word sequential. But we've only one layer. So you have a single neuron.

we can define the shape of what's input to the neural network in the first and in this case the only layer, and you can see that our input shape is super simple. It's just one value.</p>



```
model = tf.keras.Sequential(
   [keras.layers.Dense(units=1,input_shape=[1])]
   )
```

<p>As we know neural network uses a set of weights and biases to learn the relationship between given data "input" and predict "output".At the start of training, the weights are initialized with random values. This is typically done to break any symmetry in the network which refers to a situation where all the neurons in a given layer have the same weights and biases.</p>
<p>This can be a problem because it leads to the same output from each neuron, which doesn't help the network to learn anything useful and the network may struggle to learn anything meaningful and get stuck in a suboptimal solution.</p>

<p>So initializing random weight and bias helps network converge to a good solution.Let's see what does the initial value tensorflow assigned to weight and bias.</p>



```
init_weights, init_bias = model.layers[0].get_weights()

print("Initial weights:", init_weights)
print("Initial bias:", init_bias)
```

## Loss Function and Optimizer

<p>There are two function roles that you should be aware of though and these are loss functions and optimizers. </p>

<p>During training, the weights are updated in response to the errors between the model's predicted output and the actual output. This process is known as backpropagation.The optimization algorithm used during training determines how the weights are updated. The goal of the optimization algorithm is to minimize the loss function, which measures the difference between the model's predictions and the actual output. The optimizer adjusts the weights in such a way that the loss is minimized, improving the accuracy of the model's predictions.</p>

<p>So in this example we are having X and Y with relationship 2X-1(2X minus 1) which can be done manually by looking at the numbers. But the neural network has no idea of the relationship between X and Y, so it makes a guess. Say it guesses Y equals 10X-10(10X minus 10). It will then use the data that it knows about, that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was. The loss function measures this and then gives the data to the optimizer which figures out the next guess.
</p>

<p>So the optimizer thinks about how good or how badly the guess was done using the data from the loss function. Then the logic is that each guess should be better than the one before. As the guesses get better and better, an accuracy approaches 100 percent, the term convergence is used.</p>

<p>In this case, the loss is mean squared error and the optimizer is SGD which stands for stochastic gradient descent.This code defines them.</p>

```
model.compile(optimizer="sgd",loss="mean_squared_error")
```

## Visualize the model Architecture

```
tf.keras.utils.plot_model(model, show_shapes=True,show_dtype=True)
```

![Network image](/assets/img/post_imgs/neural_network_basics/TF_single_neural_network.png)

### Data to pass through model

```
xs=np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0],dtype=float)
ys=np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0],dtype=float)
```

## Training Neural network

<p>The epochs equals 500 value means that it will go through the training loop 500 times. This training loop is what we described earlier. Make a guess, measure how good or how bad the guesses with the loss function, then use the optimizer and the data to make another guess and repeat this.</p>

<p>The process of updating the weights continues until the model's performance on the training set reaches a satisfactory level or until a predetermined number of iterations has been reached in this case 500 iterations.</p>

```
model_training = model.fit(xs,ys,epochs=500,callbacks=[tensorboard_callback])
```

### Updated Weights and bias

<p>Let's view the updated weight and bias after Training the Neural Network</p>

```
fin_weights, fin_bias = model.layers[0].get_weights()

print("Final weights:", fin_weights)
print("Final bias:", fin_bias)
```

### Let's Visualize the mean squared loss in traning

```
# Plot history: MAE
plt.plot(model_training.history['loss'], label='mean_squared_error (training data)')
plt.title('mean_squared_error for relationship between x and y')
plt.ylabel('mean_squared_error value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
```

## Prediction

<p>The neural network doesn't "know" the term 2x-1 or any other mathematical equation that may describe the relationship between the inputs and outputs. Its sole objective is to adjust the weights and biases in such a way that the predicted outputs are as close as possible to the actual outputs during training, and then generalize well to unseen data during inference/prediciton.</p>

<p> After training a neural network, network use the final weights that were learned during training to make predictions on new, unseen data.Let's run the code and see what is the predicted output</p>

<p>When the model has finished training, it will then give you back values using the predict method. So it hasn't previously seen 10, and what do you think it will return when you pass it a 10? Now you might think it would return 19 because after all Y equals 2X minus 1, and you think it should be 19.Let's see the predicted output</p>

```
print(model.predict([10.0]))
```

<p>The preicted value is very close to 19 but not exactly 19.</p>

<p>Now why do you think that would be? Ultimately there are two main reasons.</p>

1. The first is that you trained it using very little data. There's only six points. Those six points are linear but there's no guarantee that for every X, the relationship will be Y equals 2X minus 1. There's a very high probability that Y equals 19 for X equals 10, but the neural network isn't positive. So it will figure out a realistic value for Y.

2. That's the second main reason. When using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit.

## Different approach by machine learning and deep learning for this problem:

<p>In this type of regression machine learning approach, the data scientist needs to manually select the appropriate ***model architecture***, ***feature engineering***, and ***hyperparameters tuning*** to achieve the best performance.</p>

<p>The model architecture could be a simple linear regression model, a polynomial regression model, or any other model that is appropriate for the data at hand. The feature engineering step involves selecting the most relevant features from the input data and transforming them into a format that the model can use. The hyperparameters tuning step involves selecting the best values for the model's hyperparameters to optimize its performance.</p>

<p>On the other hand, the TensorFlow single neuron approach ***automates*** many of these steps, such as ***model architecture selection and hyperparameter tuning***. The data scientist only needs to specify the *number of neurons* in the layer, the *activation function*, and the *loss function*. TensorFlow will take care of the rest, including selecting the appropriate weight and bias values and optimizing them using backpropagation.<p>

<p>In general, for this kind of regression problem with a single input and a single output, both machine learning and deep learning approaches can work well. The choice between the two depends on several factors, such as the amount and quality of data, the complexity of the problem, and the expertise and resources available to the data scientist.</p>
<p>If the problem is relatively simple and the data is not too large, a regression machine learning approach may be sufficient. If the problem is more complex or the data is very large, a deep learning approach may be more suitable.</p>

## General Difference between approach in DL and ML

- **Representation of the model**: In deep learning, the model is represented as a neural network, which is a hierarchical structure composed of multiple layers of interconnected neurons. Each neuron computes a weighted sum of its inputs, applies an activation function to the result, and passes the output to the next layer. In contrast, traditional machine learning models are often represented as a mathematical function that maps input variables to output variables.

- **Feature engineering**: In traditional machine learning, feature engineering is often a crucial step in the modeling process. This involves selecting and transforming the input variables to create new features that are more informative for the modeling task. In deep learning, the neural network is often able to automatically learn useful features from the raw input data, without the need for explicit feature engineering.

- **Training process**: In deep learning, the training process typically involves defining a loss function that measures the difference between the predicted output of the model and the true output, and then using an optimization algorithm (such as stochastic gradient descent) to minimize the loss by adjusting the weights and biases of the neural network. In traditional machine learning, the training process may involve different algorithms depending on the type of model used (e.g., linear regression, decision trees, support vector machines, etc.).

- **Model capacity**: In deep learning, the neural network can have a very large number of parameters (i.e., weights and biases) that can be tuned during training. This allows the model to capture complex patterns and relationships in the data. In traditional machine learning, the model capacity is often more limited, and may depend on the number and complexity of the features used.

- **Generalization performance**: In deep learning, there is a risk of overfitting the model to the training data, especially if the model capacity is large and the amount of training data is limited. To address this, various regularization techniques (such as dropout, weight decay, and early stopping) can be used. In traditional machine learning, overfitting can also be a concern, and various regularization techniques can also be applied (such as L1/L2 regularization, pruning, and ensembling).

## Conclusion
<p>Overall, we covered the basics of a single neuron neural network, including its architecture, training process, and prediction, as well as the differences between Machine learning and Deep Learning approach. These concepts provide a solid foundation for learning more about neural networks and their applications in machine learning and deep learning.</p>

### Reference:

-[From Deeplearning.AI cousera course](https://www.coursera.org/learn/introduction-tensorflow)