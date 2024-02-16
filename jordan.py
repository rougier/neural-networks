#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Jordan recurrent network
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid activation function '''
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    ''' Derivative of the sigmoid function '''
    return x * (1 - x)

class JordanNetwork:
    ''' Jordan network implementation '''

    def __init__(self, *layer_sizes):
        ''' Initialize the Jordan network with given layer sizes '''

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.num_layers - 1)]

    def feedforward(self, input_data):
        ''' Perform feedforward propagation '''

        activations = [input_data]
        for i in range(self.num_layers - 1):
            weighted_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(sigmoid(weighted_input))
        return activations

    def backpropagate(self, input_data, target, learning_rate=0.1):
        ''' Perform backpropagation '''

        activations = self.feedforward(input_data)
        deltas = [None] * (self.num_layers - 1)

        # Compute error on output layer
        output_error = target - activations[-1]
        deltas[-1] = output_error * sigmoid_derivative(activations[-1])

        # Backpropagate errors
        for i in range(self.num_layers - 2, 0, -1):
            deltas[i-1] = np.dot(deltas[i], self.weights[i].T) * sigmoid_derivative(activations[i])

        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] += learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0)

    def train(self, training_data, epochs=1000, learning_rate=0.1):
        ''' Train the Jordan network '''

        for epoch in range(epochs):
            for input_data, target in training_data:
                self.backpropagate(input_data, target, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss {self.compute_loss(training_data)}')

    def compute_loss(self, training_data):
        ''' Compute total loss on the training data '''

        total_loss = 0
        for input_data, target in training_data:
            activations = self.feedforward(input_data)
            output_error = target - activations[-1]
            total_loss += np.sum(output_error ** 2)
        return total_loss

if __name__ == '__main__':
    # Example: Learning a simple time series
    training_data = [
        (np.array([[1, 0, 0, 0]]), np.array([[0, 1, 0, 0]])),
        (np.array([[0, 1, 0, 0]]), np.array([[0, 0, 1, 0]])),
        (np.array([[0, 0, 1, 0]]), np.array([[0, 0, 0, 1]])),
        (np.array([[0, 0, 0, 1]]), np.array([[0, 0, 1, 0]])),
        (np.array([[0, 0, 1, 0]]), np.array([[0, 1, 0, 0]])),
        (np.array([[0, 1, 0, 0]]), np.array([[1, 0, 0, 0]]))
    ]

    jordan_network = JordanNetwork(4, 8, 4)
    jordan_network.train(training_data, epochs=5000, learning_rate=0.1)

    # Test the trained network
    for input_data, target in training_data:
        output = jordan_network.feedforward(input_data)[-1]
        print(f'Input: {input_data}, Target: {target}, Output: {output}')