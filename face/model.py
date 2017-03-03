from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_part as mp


class Generator:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim

    def inference(self, z, reuse=False):
        # linear projection.
        z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 7 * 7, reuse=reuse, with_w=True)
        # reshape for cnn inputs.
        h0 = tf.reshape(z_, [-1, 7, 7, self.first_conv_dim * 8])
        # batch norm
        h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse))

        # deconv1 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
        deconv_h1, deconv_h1_w, deconv_h1_b = mp.conv2d_transpose('g_deconv_h1', h0,
                                                                  [5, 5, self.first_conv_dim * 4, h0.get_shape()[-1]],
                                                                  [self.batch_size, 14, 14, self.first_conv_dim * 4],
                                                                  [self.first_conv_dim * 4], [1, 2, 2, 1],
                                                                  padding='SAME', reuse=reuse,  with_w=True)
        h1 = tf.nn.relu(mp.batch_norm(deconv_h1, scope_name='g_bn_h1', reuse=reuse))

        # deconv2 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
        deconv_h2, deconv_h2_w, deconv_h2_b = mp.conv2d_transpose('g_deconv_h2', h1,
                                                                  [5, 5, 3, h1.get_shape()[-1]],
                                                                  [self.batch_size, 28, 28, 3],
                                                                  [3], [1, 2, 2, 1],
                                                                  padding='SAME', reuse=reuse, with_w=True)
        print(deconv_h2.get_shape())
        return tf.nn.tanh(deconv_h2)


class Descriminator:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim

    def inference(self, x, reuse=False):
        print("===")
        print(x.get_shape())
        # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
        conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('d_conv_h0', x,
                                                  [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                  [self.first_conv_dim],
                                                  [1, 2, 2, 1],
                                                  padding='SAME', reuse=reuse, with_w=True)
        h0 = mp.lrelu(conv_h0)
        print(h0.get_shape())
        # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
        conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('d_conv_h1', h0,
                                                  [5, 5, h0.get_shape()[-1], self.first_conv_dim],
                                                  [self.first_conv_dim],
                                                  [1, 2, 2, 1],
                                                  padding='SAME', reuse=reuse, with_w=True)
        h1 = mp.lrelu(conv_h1)

        # linear projection
        h2 = mp.linear_project('d_lin_project_h1', tf.reshape(h1, [self.batch_size, 7*7*64]), 1, reuse=reuse)
        return tf.nn.sigmoid(h2), h2


if __name__ == '__main__':
    g = Generator(10, 5)
    z = tf.placeholder(tf.float32, [None, 100], name='z')
    g.inference(z)

    d = Descriminator(10, 5)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    d.inference(x)
