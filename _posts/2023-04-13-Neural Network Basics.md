---
layout: post
title: Neural Networks Basics
date: 2023-04-13
categories: Neural Networks
tags: [Neural Networks, Deep Learning]
---

![header image](\assets\img\post_imgs\neural network basics\header.jpg)

## Intoduciton:

<p> A neural network is a computer program that tries to learn from data, similar to how humans learn from experience. It is made up of a series of interconnected nodes, or neurons, that work together to solve a specific task. Each neuron receives inputs from other neurons or directly from the data, and uses a set of weights and biases to process the information and produce an output.Once trained, the network can be used to make predictions or decisions based on new input data.In simple terms a neural network is basically a set of functions which can learn to recognize and generalize patterns in data,making them a powerful tool for a wide range of machine learning applications from image recognition and speech recognition to autonomous vehicles and natural language processing.</p>

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

## Creating the model

<p>The simplest possible neural network is one that has only one neuron in it, and that's what this line of code does.

In keras, you use the word dense to define a layer of connected neurons. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron.

Successive layers are defined in sequence, hence the word sequential. But we've only one layer. So you have a single neuron.

we can define the shape of what's input to the neural network in the first and in this case the only layer, and you can see that our input shape is super simple. It's just one value.</p>

### Reference:

-[From Deeplearning.AI cousera course](https://www.coursera.org/learn/introduction-tensorflow)

```
model = tf.keras.Sequential(
   [keras.layers.Dense(units=1,input_shape=[1])]
   )
```

<p>As we know neural network uses a set of weights and biases to learn the relationship between given data "input" and predict "output".At the start of training, the weights are initialized with random values. This is typically done to break any symmetry in the network which refers to a situation where all the neurons in a given layer have the same weights and biases. This can be a problem because it leads to the same output from each neuron, which doesn't help the network to learn anything useful and the network may struggle to learn anything meaningful and get stuck in a suboptimal solution.

So initializing random weight and bias helps network converge to a good solution.Let's see what does the initial value tensorflow assigned to weight and bias.

</p>

```
init_weights, init_bias = model.layers[0].get_weights()

print("Initial weights:", init_weights)
print("Initial bias:", init_bias)
```

## Loss Function and Optimizer

<p>There are two function roles that you should be aware of though and these are loss functions and optimizers. 
During training, the weights are updated in response to the errors between the model's predicted output and the actual output. This process is known as backpropagation.The optimization algorithm used during training determines how the weights are updated. The goal of the optimization algorithm is to minimize the loss function, which measures the difference between the model's predictions and the actual output. The optimizer adjusts the weights in such a way that the loss is minimized, improving the accuracy of the model's predictions.

So in this example we are having X and Y with relationship 2X-1(2X minus 1) which can be done manually by looking at the numbers. But the neural network has no idea of the relationship between X and Y, so it makes a guess. Say it guesses Y equals 10X-10(10X minus 10). It will then use the data that it knows about, that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was. The loss function measures this and then gives the data to the optimizer which figures out the next guess.

So the optimizer thinks about how good or how badly the guess was done using the data from the loss function. Then the logic is that each guess should be better than the one before. As the guesses get better and better, an accuracy approaches 100 percent, the term convergence is used.

In this case, the loss is mean squared error and the optimizer is SGD which stands for stochastic gradient descent.This code defines them.

</p>

```
model.compile(optimizer="sgd",loss="mean_squared_error")
```

## Visualize the model Architecture

```
tf.keras.utils.plot_model(model, show_shapes=True,show_dtype=True)
```

![Network image](assets\img\post_imgs\neural network basics\TF_single_neural_network.png)

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

<p>When the model has finished training, it will then give you back values using the predict method. So it hasn't previously seen 10, and what do you think it will return when you pass it a 10? Now you might think it would return 19 because after all Y equals 2X minus 1, and you think it should be 19. But when you try this in the workbook yourself, you'll see that it will return a value very close to 19 but not exactly 19.

Now why do you think that would be? Ultimately there are two main reasons.

The first is that you trained it using very little data. There's only six points. Those six points are linear but there's no guarantee that for every X, the relationship will be Y equals 2X minus 1. There's a very high probability that Y equals 19 for X equals 10, but the neural network isn't positive. So it will figure out a realistic value for Y.

That's the second main reason. When using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit.</p>

```
print(model.predict([10.0]))
```

<p>

niyutkavi18@gmail.com
so each neuron has one weight and bias ?
No, each neuron can have multiple weights and biases, depending on the number of inputs and the architecture of the neural network. In fact, in a typical neural network, each neuron in a layer has a weight associated with each input from the previous layer, and a bias term. The number of weights and biases associated with a neuron depends on the number of inputs it receives.

niyutkavi18@gmail.com
if i have 1 input and 1 neuron how many weight and bias are there?
If you have 1 input and 1 neuron, then there will be 1 weight and 1 bias associated with that neuron. The weight represents the strength of the connection between the input and the neuron, while the bias represents the neuron's threshold for activation.

niyutkavi18@gmail.com
multiple neurons are called layers?
Yes, multiple neurons are often organized in layers in a neural network. A layer is a collection of neurons that process input data together, and each neuron in a layer is connected to every neuron in the previous layer (if there is one) and to every neuron in the following layer (if there is one). The output of one layer becomes the input of the next layer, forming a pipeline of computations that gradually transform the input data into the desired output.

niyutkavi18@gmail.com
i have single input and layer of 4 neuron , now how many weight and bias i have
If you have a single input and a layer of 4 neurons, you will need 4 weights (one weight for each connection between the input and each neuron in the layer) and 4 biases (one bias for each neuron in the layer). So the total number of weights and biases in this case will be 8.

niyutkavi18@gmail.com
can you quiz me like the request i made previously like if single input layer and 4 neuron layer how many weight and bias ...i need to better understand this concept
Sure, I can give you some questions to help you practice:

If you have a single input and a single neuron, how many weights and biases are there?
If you have a single input and a layer of 4 neurons, how many weights and biases are there in total?
If you have a layer of 10 neurons and each neuron is connected to the previous layer with 50 inputs, how many weights and biases are there in total?
If you have a neural network with 3 layers, where the first layer has 50 neurons, the second layer has 100 neurons, and the third layer has 10 neurons, how many weights and biases are there in total?
I hope these questions help you better understand the concept of weights and biases in neural networks!

</p>
