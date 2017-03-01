from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_part as mp

class Generator:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim

    def inference(self, z):
        # linear projection.
        z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 4 * 4, with_w=True)
        # reshape for cnn inputs.
        h0 = tf.reshape(z_, [-1, 4, 4, self.first_conv_dim * 8])
        # batch norm
        h0 = tf.nn.relu(mp.batch_norm(h0))


if __name__ == '__main__':
    g = Generator(10, 5)
    z = tf.placeholder(tf.float32, [None, 100], name='z')
    g.inference(z)