from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class Distribution:
    '''
    Distribution class
    '''
    def __init__(self, mu, sigma):
        '''
        :param mu:
        :param sigma: variance of distribution
        '''
        self.mu = mu
        self.sigma = sigma

    def dist(self, xs=np.linspace(-5,5,1000)):
        dist = norm.pdf(xs, loc=self.mu, scale=self.sigma)
        return dist

    def show(self):
        xs = np.linspace(-5, 5, 1000)
        dist = self.dist(xs)
        plt.plot(xs, dist)
        plt.show()

if __name__ == '__main__':
    dist = Distribution(-1, 1)
    dist.show()