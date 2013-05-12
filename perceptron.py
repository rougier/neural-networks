#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    return x > 0

class Perceptron:
    ''' Perceptron class. '''

    def __init__(self, n, m):
        ''' Initialization of the perceptron with given sizes.  '''

        self.input  = np.ones(n+1)
        self.output = np.ones(m)
        self.weights= np.zeros((m,n+1))
        self.reset()

    def reset(self):
        ''' Reset weights '''

        Z = np.random.random(self.weights.shape)
        self.weights[...] = (2*Z-1)*.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer (but not bias)
        self.input[1:]  = data
        self.output[...] = sigmoid(np.dot(self.weights,self.input))

        # Return output
        return self.output

    def propagate_backward(self, target, lrate=0.1):
        ''' Back propagate error related to target using lrate. '''

        error = np.atleast_2d(target-self.output)
        input = np.atleast_2d(self.input)
        self.weights += lrate*np.dot(error.T,input)

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def learn(network,samples, epochs=250, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print i, samples['input'][i], '%.2f' % o[0],
            print '(expected %.2f)' % samples['output'][i]
        print
        

    network = Perceptron(2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print "Learning the OR logical function"
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print "Learning the AND logical function"
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 0
    samples[2] = (0,1), 0
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print "Failed at learning the XOR logical function"
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)
