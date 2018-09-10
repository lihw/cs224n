#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    z1 = np.matmul(X, W1) + b1
    a1 = sigmoid(z1) # b1 will be added to each row of XW1
    z2 = np.matmul(a1, W2) + b2
    a2 = softmax(z2) # b2 will be added to each row of L1W2
    CE = -np.log(a2) * labels
    cost = np.sum(CE)
    #raise NotImplementedError
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradb2 = np.zeros([1, Dy])
    gradW2 = np.zeros([H, Dy])
    gradb1 = np.zeros([1, H])
    gradW1 = np.zeros([Dx, H])

    gradCE = a2 - labels

    M = X.shape[0]
        
    for i in range(M):
        gradb2_i = gradCE[i] # 1 x Dy
        gradb2 += gradb2_i

        gradW2_i = np.outer(a1[i], gradb2_i) # H x Dy
        gradW2 += gradW2_i

        gradb1_i = np.matmul(gradb2_i, W2.T) * sigmoid_grad(z1[i]) # 1 x H
        gradb1 += gradb1_i

        gradW1_i = np.outer(X[i], gradb1_i) # Dx x H
        gradW1 += gradW1_i

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
