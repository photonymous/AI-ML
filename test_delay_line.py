# This file tests the idea of using a delay line in a neural network implemented in Pytorch.
# First, I will type all of the code comments describing every step of the process.
# Then I will type the code.
# It is important to look at the gradient of the input sub-network's weights with respect to the 
# system output, and compare between the two models.


# Import the libraries we will need:
import torch
import torch.nn as nn
import numpy as np
import sys

# Lets define the 4 input samples that will be used by both models:
x1 = torch.tensor([1.1,1.2,1.3,1.4], requires_grad=True)

#===============================================================================
# Model 1: With delay line
# The first model will show how the delay line works.
# There are 4 input data samples (floating point numbers)
# It will process one input sample at a time.
# It will feed these thorugh a "sub-network" consisting of a linear layer, a tanh non-linear activation, and a weighted summation to output a single floating point number.
# This will be pushed into a 4-element delay line.
# All elements of the delay line will feed the second sub-network, which will be similar to the first sub-network.
# Eventually, the delay line will be populated with all 4 outputs of the first sub-network, representing the 4 processed samples.
# The second sub-network's output with this input will be the final output.

# Define the first sub-network using Pytorch primitives (tensors, parameters, etc):
# The first layer is a linear layer taking one input with 4 neurons, applying 4 tanh nonlinearities, and outputting 4 values.
# Here are those weights and biases
w1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
b1 = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)

# Then it projects the output back to a single number using these weights:
w2 = torch.tensor([[1.5],[2.6],[3.7],[4.8]], requires_grad=True)

# Here's the delay line:
delay_line = torch.zeros(4, requires_grad=True)

# And now the second sub-network:
# The first layer is a linear layer taking 4 inputs with 4 neurons, applying 4 tanh nonlinearities, and outputting 4 values.
# Here are those weights and biases (pseudo random 1-decimal place numbers):
w3 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                   [0.5, 0.6, 0.7, 0.8],
                   [0.9, 1.0, 1.1, 1.2],
                   [1.3, 1.4, 1.5, 1.6]], requires_grad=True)
b3 = torch.tensor([0.9, 0.8, 0.7, 0.6], requires_grad=True)

# Then it projects the output back to a single number using these weights:
w4 = torch.tensor([[1.8],[2.7],[3.6],[4.5]], requires_grad=True)







#===============================================================================
# Model 2: No delay line
# The second model shows how this exact same process can be implemented without the delay line.
# In this case, four copies of the sub-network will be used, and the outputs of each sub-network will be fed to the second sub-network.
# All 4 of thefirst sub-networks' instances share weights, so perhaps this can be implemented as a convolutional neural network.
# The second sub-network will have 4 inputs, and will output a single floating point number.






#===============================================================================
# Run Model 1:
# Now we will run the first model and we will examine the gradients of the weights of the first sub-network with respect to the output of the second sub-network.
# Lets do this in a for loop to process one sample at a time from x1:
for i in range(4):
    # Print the model number and the sample index:
    print("Model 1:" + " i = " + str(i))
    # First, we need to compute the output of the first sub-network:
    y1 = torch.matmul(x1[i], w1.t()) + b1
    y1 = torch.tanh(y1)
    y1 = torch.matmul(y1, w2)
    # Now we need to push this output into the delay line:
    delay_line = torch.roll(delay_line, 1)
    delay_line[0] = y1
    # Now we need to run the second sub-network:
    # First, we need to compute the output of the second sub-network:
    y2 = torch.matmul(delay_line, w3.t()) + b3
    y2 = torch.tanh(y2)
    y2 = torch.matmul(y2, w4)
    print("y2 = " + str(y2))
    # Now we need to compute the loss. We don't have any training data, so we can just pretend 0 is the target:
    loss = (y2 - 0)**2
    # Now we need to compute the gradients of the weights of the first sub-network with respect to the loss:
    loss.backward()
    # Now we need to print the gradients of the weights of the first sub-network with respect to the loss:
    print("Gradients of the weights of the first sub-network with respect to the loss:")
    print(w1.grad)
    print(b1.grad)
    print(w2.grad)
    print("")
    # Now we need to print the gradients of the weights of the second sub-network with respect to the loss:
    print("Gradients of the weights of the second sub-network with respect to the loss:")
    print(w3.grad)
    print(b3.grad)
    print(w4.grad)
    print("")
    # Now we need to zero out the gradients of the weights of the first sub-network:
    w1.grad.zero_()
    b1.grad.zero_()
    w2.grad.zero_()
    # Now we need to zero out the gradients of the weights of the second sub-network:
    w3.grad.zero_()
    b3.grad.zero_()
    w4.grad.zero_()






    







