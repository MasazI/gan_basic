from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


def load_datas():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


if __name__ == '__main__':
    mnist = load_datas()
    for i in range(3):
        batch = mnist.train.next_batch(3)
        images = batch[0].reshape([3, 28, 28])
        print(images.shape)
        plt.title('Label is {label}'.format(label=batch[1][0]))
        plt.imshow(images[0], cmap='gray')
        plt.show()

