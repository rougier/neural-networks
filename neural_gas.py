#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Neural gas
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

class NeuralGas:
    ''' Neural gas '''

    def __init__(self, *args):
        ''' Initialize neural gas '''

        self.codebook = np.zeros(args)
        self.reset()


    def reset(self):
        ''' Reset weights '''

        self.codebook = np.random.random(self.codebook.shape)


    def learn(self, samples, epochs=10000, sigma=(10, 0.001), lrate=(0.5,0.005)):
        ''' Learn samples '''

        sigma_i, sigma_f = sigma
        lrate_i, lrate_f = lrate

        for i in range(epochs):
            # Adjust learning rate and neighborhood
            t = i/float(epochs)
            lrate = lrate_i*(lrate_f/float(lrate_i))**t
            sigma = sigma_i*(sigma_f/float(sigma_i))**t

            # Get random sample
            index = np.random.randint(0,samples.shape[0])
            data = samples[index]

            # Compute distances to data 
            D = ((self.codebook-data)**2).sum(axis=-1).flatten()

            # Get ordered distance indices
            I = np.argsort(np.argsort(D))

            # Compute h(k/sigma)
            H = np.exp(-I/sigma).reshape(self.codebook.shape[:-1])

            # Move nodes towards data according to H
            delta = data - self.codebook
            for i in range(self.codebook.shape[-1]):
                self.codebook[...,i] += lrate * H * delta[...,i]



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    try:    from voronoi import voronoi
    except: voronoi = None

    def learn(network, samples, epochs=25000, sigma=(10, 0.01), lrate=(0.5,0.005)):
        network.learn(samples, epochs)

        fig = plt.figure(figsize=(10,10))
        axes = fig.add_subplot(1,1,1)
        # Draw samples
        x,y = samples[:,0], samples[:,1]
        plt.scatter(x, y, s=1.0, color='b', alpha=0.1, zorder=1)
        # Draw network
        x,y = network.codebook[...,0], network.codebook[...,1]
        plt.scatter (x, y, s=50, c='w', edgecolors='k', zorder=3)
        if voronoi is not None:
            segments = voronoi(x.ravel(),y.ravel())
            lines = matplotlib.collections.LineCollection(segments, color='0.65')
            axes.add_collection(lines)
        plt.axis([0,1,0,1])
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Example 1: uniform distribution (2d)
    # -------------------------------------------------------------------------
    print 'NG over two-dimensional uniform square'
    gas = NeuralGas(10,10,2)
    samples = np.random.random((10000,2))
    learn(gas,samples)

    # Example 2: non-uniform distribution (2d)
    # -------------------------------------------------------------------------
    print 'NG over two-dimensional non-uniform disc'
    gas = NeuralGas(10,10,2)
    samples = np.random.normal(loc=.5, scale=.2,size=(10000,2))
    learn(gas,samples)

    # Example 3: non-uniform disc distribution (2d)
    # -------------------------------------------------------------------------
    print 'NG over two-dimensional non-uniform ring'
    gas = NeuralGas(10,10,2)
    angles = np.random.random(10000)*2*np.pi
    radius = 0.25+np.random.random(10000)*.25
    samples = np.zeros((10000,2))
    samples[:,0] = 0.5+radius*np.cos(angles)
    samples[:,1] = 0.5+radius*np.sin(angles)
    learn(gas,samples)
