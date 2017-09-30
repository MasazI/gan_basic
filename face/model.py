from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_part as mp


class GeneratorExpand:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.fc_dim = 8192

        self.g_bn0 = mp.BatchNorm(name='g_bn0')
        self.g_bn1 = mp.BatchNorm(name='g_bn1')
        self.g_bn2 = mp.BatchNorm(name='g_bn2')
        self.g_bn3 = mp.BatchNorm(name='g_bn3')
        self.g_bn1_2 = mp.BatchNorm(name='g_bn1_2')
        self.g_bn2_2 = mp.BatchNorm(name='g_bn2_2')
        self.g_bn3_2 = mp.BatchNorm(name='g_bn3_2')

    def sampler(self, z, reuse=True, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse, trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 8])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # deconv1 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h1, deconv_h1_w, deconv_h1_b = mp.conv2d_transpose('g_deconv_h1', h0,
                                                                      [5, 5, self.first_conv_dim * 8, h0.get_shape()[-1]],
                                                                      [self.batch_size, 8, 8, self.first_conv_dim * 8],
                                                                      [self.first_conv_dim * 8], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse,  with_w=True)
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))
            # h1 = tf.nn.relu(mp.batch_norm(deconv_h1, scope_name='g_bn_h1', reuse=reuse, trainable=trainable))

            # 1st conv
            conv_h1_2, conv_h1_2_w, conv_h1_2_b = mp.conv2d('g_conv_h1_2', h1,
                                                            [5, 5, h1.get_shape()[-1], self.first_conv_dim * 8],
                                                            [self.first_conv_dim * 8],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h1_2 = tf.nn.relu(self.g_bn1_2(conv_h1_2, trainable=trainable))

            # deconv2 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h2, deconv_h2_w, deconv_h2_b = mp.conv2d_transpose('g_deconv_h2', h1_2,
                                                                      [5, 5, self.first_conv_dim * 4, h1_2.get_shape()[-1]],
                                                                      [self.batch_size, 16, 16, self.first_conv_dim * 4],
                                                                      [self.first_conv_dim * 4], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            print(deconv_h2.get_shape())
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))
            # h2 = tf.nn.relu(mp.batch_norm(deconv_h2, scope_name='g_bn_h2', reuse=reuse, trainable=trainable))

            # 2nd conv
            conv_h2_2, conv_h2_2_w, conv_h2_2_b = mp.conv2d('g_conv_h2_2', h2,
                                                            [5, 5, h2.get_shape()[-1], self.first_conv_dim * 4],
                                                            [self.first_conv_dim * 4],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h2_2 = tf.nn.relu(self.g_bn2_2(conv_h2_2, trainable=trainable))

            # 3rd
            deconv_h3, deconv_h3_w, deconv_h3_b = mp.conv2d_transpose('g_deconv_h3', h2_2,
                                                                      [5, 5, self.first_conv_dim * 2, h2_2.get_shape()[-1]],
                                                                      [self.batch_size, 32, 32, self.first_conv_dim * 2],
                                                                      [self.first_conv_dim * 2], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))
            # h3 = tf.nn.relu(mp.batch_norm(deconv_h3, scope_name='g_bn_h3', reuse=reuse, trainable=trainable))

            # 3rd conv
            conv_h3_2, conv_h3_2_w, conv_h3_2_b = mp.conv2d('g_conv_h3_2', h3,
                                                            [5, 5, h3.get_shape()[-1], self.first_conv_dim * 2],
                                                            [self.first_conv_dim * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h3_2 = tf.nn.relu(self.g_bn3_2(conv_h3_2, trainable=trainable))


            # 4th
            deconv_h4, deconv_h4_w, deconv_h4_b = mp.conv2d_transpose('g_deconv_h4', h3_2,
                                                                      [5, 5, 3, h3_2.get_shape()[-1]],
                                                                      [self.batch_size, 64, 64, 3],
                                                                      [3], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            return tf.nn.tanh(deconv_h4)


    def inference(self, z, reuse=False, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse,
                                       trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 8])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # deconv1 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h1, deconv_h1_w, deconv_h1_b = mp.conv2d_transpose('g_deconv_h1', h0,
                                                                      [5, 5, self.first_conv_dim * 8, h0.get_shape()[-1]],
                                                                      [self.batch_size, 8, 8, self.first_conv_dim * 8],
                                                                      [self.first_conv_dim * 8], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse,  with_w=True, trainable=trainable)
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))
            # h1 = tf.nn.relu(mp.batch_norm(deconv_h1, scope_name='g_bn_h1', reuse=reuse, trainable=trainable))

            # 1st conv
            conv_h1_2, conv_h1_2_w, conv_h1_2_b = mp.conv2d('g_conv_h1_2', h1,
                                                            [5, 5, h1.get_shape()[-1], self.first_conv_dim * 8],
                                                            [self.first_conv_dim * 8],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h1_2 = tf.nn.relu(self.g_bn1_2(conv_h1_2, trainable=trainable))

            # deconv2 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h2, deconv_h2_w, deconv_h2_b = mp.conv2d_transpose('g_deconv_h2', h1_2,
                                                                      [5, 5, self.first_conv_dim * 4,
                                                                       h1_2.get_shape()[-1]],
                                                                      [self.batch_size, 16, 16,
                                                                       self.first_conv_dim * 4],
                                                                      [self.first_conv_dim * 4], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            print(deconv_h2.get_shape())
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))
            # h2 = tf.nn.relu(mp.batch_norm(deconv_h2, scope_name='g_bn_h2', reuse=reuse, trainable=trainable))

            # 2nd conv
            conv_h2_2, conv_h2_2_w, conv_h2_2_b = mp.conv2d('g_conv_h2_2', h2,
                                                            [5, 5, h2.get_shape()[-1], self.first_conv_dim * 4],
                                                            [self.first_conv_dim * 4],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h2_2 = tf.nn.relu(self.g_bn2_2(conv_h2_2, trainable=trainable))

            # 3rd
            deconv_h3, deconv_h3_w, deconv_h3_b = mp.conv2d_transpose('g_deconv_h3', h2_2,
                                                                      [5, 5, self.first_conv_dim * 2,
                                                                       h2_2.get_shape()[-1]],
                                                                      [self.batch_size, 32, 32,
                                                                       self.first_conv_dim * 2],
                                                                      [self.first_conv_dim * 2], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))
            # h3 = tf.nn.relu(mp.batch_norm(deconv_h3, scope_name='g_bn_h3', reuse=reuse, trainable=trainable))

            # 3rd conv
            conv_h3_2, conv_h3_2_w, conv_h3_2_b = mp.conv2d('g_conv_h3_2', h3,
                                                            [5, 5, h3.get_shape()[-1], self.first_conv_dim * 2],
                                                            [self.first_conv_dim * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            h3_2 = tf.nn.relu(self.g_bn3_2(conv_h3_2, trainable=trainable))

            # 4th
            deconv_h4, deconv_h4_w, deconv_h4_b = mp.conv2d_transpose('g_deconv_h4', h3_2,
                                                                      [5, 5, 3, h3_2.get_shape()[-1]],
                                                                      [self.batch_size, 64, 64, 3],
                                                                      [3], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)

            return tf.nn.tanh(deconv_h4)


class GeneratorResnet:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.fc_dim = self.first_conv_dim * 9 * 4 * 4

        self.g_bn0 = mp.BatchNorm(name='g_bn0')
        self.g_bn1 = mp.BatchNorm(name='g_bn1')
        self.g_bn2 = mp.BatchNorm(name='g_bn2')
        self.g_bn3 = mp.BatchNorm(name='g_bn3')

    def sampler(self, z, reuse=True, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G-s")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 9 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse, trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 9])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # TODO step 1 change residual blocks layer x 8

            # TODO step 2 change subpixel convolution layer x 4

            # 1st
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('g_conv_h1', h0,
                                                      [3, 3, h0.get_shape()[-1], self.first_conv_dim * 9 * 2 * 2],
                                                      [self.first_conv_dim * 9 * 2 * 2],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            deconv_h1 = mp.subpixel_conv(conv_h1, 2, output_c=self.first_conv_dim * 9)
            print(deconv_h1.get_shape())
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))

            # 2nd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('g_conv_h2', h1,
                                                            [5, 5, h1.get_shape()[-1], self.first_conv_dim * 6 * 2 * 2],
                                                            [self.first_conv_dim * 6 * 2 * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            deconv_h2 = mp.subpixel_conv(conv_h2, 2, output_c=self.first_conv_dim * 6)
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))

            # 3rd
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('g_conv_h3', h2,
                                                            [5, 5, h2.get_shape()[-1], self.first_conv_dim * 3 * 2 * 2],
                                                            [self.first_conv_dim * 3 * 2 * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            deconv_h3 = mp.subpixel_conv(conv_h3, 2, output_c=self.first_conv_dim * 3)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))

            # 4th
            conv_h4, conv_h4_w, conv_4_b = mp.conv2d('g_conv_h4', h3,
                                                  [5, 5, h3.get_shape()[-1], 3 * 2 * 2],
                                                  [3 * 2 * 2],
                                                  [1, 1, 1, 1],
                                                  padding='SAME', reuse=reuse, with_w=True,
                                                  trainable=trainable)
            deconv_h4 = mp.subpixel_conv(conv_h4, 2, 3)
            print(deconv_h4.get_shape())
            return tf.nn.tanh(deconv_h4)


    def inference(self, z, reuse=False, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 9 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse,
                                       trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 9])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # 1st
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('g_conv_h1', h0,
                                                      [3, 3, h0.get_shape()[-1], self.first_conv_dim * 9 * 2 * 2],
                                                      [self.first_conv_dim * 9 * 2 * 2],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            deconv_h1 = mp.subpixel_conv(conv_h1, 2, output_c=self.first_conv_dim * 9)
            print(deconv_h1.get_shape())
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))

            # 2nd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('g_conv_h2', h1,
                                                            [5, 5, h1.get_shape()[-1], self.first_conv_dim * 6 * 2 * 2],
                                                            [self.first_conv_dim * 6 * 2 * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            deconv_h2 = mp.subpixel_conv(conv_h2, 2, output_c=self.first_conv_dim * 6)
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))

            # 3rd
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('g_conv_h3', h2,
                                                            [5, 5, h2.get_shape()[-1], self.first_conv_dim * 3 * 2 * 2],
                                                            [self.first_conv_dim * 3 * 2 * 2],
                                                            [1, 1, 1, 1],
                                                            padding='SAME', reuse=reuse, with_w=True,
                                                            trainable=trainable)
            deconv_h3 = mp.subpixel_conv(conv_h3, 2, output_c=self.first_conv_dim * 3)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))

            # 4th
            conv_h4, conv_h4_w, conv_4_b = mp.conv2d('g_conv_h4', h3,
                                                  [5, 5, h3.get_shape()[-1], 3 * 2 * 2],
                                                  [3 * 2 * 2],
                                                  [1, 1, 1, 1],
                                                  padding='SAME', reuse=reuse, with_w=True,
                                                  trainable=trainable)
            deconv_h4 = mp.subpixel_conv(conv_h4, 2, 3)
            print("subpix h4")
            print(deconv_h4.get_shape())

            return tf.nn.tanh(deconv_h4)


class Generator:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.fc_dim = 8192

        self.g_bn0 = mp.BatchNorm(name='g_bn0')
        self.g_bn1 = mp.BatchNorm(name='g_bn1')
        self.g_bn2 = mp.BatchNorm(name='g_bn2')
        self.g_bn3 = mp.BatchNorm(name='g_bn3')

    def sampler(self, z, reuse=True, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse, trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 8])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # deconv1 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h1, deconv_h1_w, deconv_h1_b = mp.conv2d_transpose('g_deconv_h1', h0,
                                                                      [5, 5, self.first_conv_dim * 8, h0.get_shape()[-1]],
                                                                      [self.batch_size, 8, 8, self.first_conv_dim * 8],
                                                                      [self.first_conv_dim * 8], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse,  with_w=True)
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))
            # h1 = tf.nn.relu(mp.batch_norm(deconv_h1, scope_name='g_bn_h1', reuse=reuse, trainable=trainable))

            # deconv2 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h2, deconv_h2_w, deconv_h2_b = mp.conv2d_transpose('g_deconv_h2', h1,
                                                                      [5, 5, self.first_conv_dim * 4, h1.get_shape()[-1]],
                                                                      [self.batch_size, 16, 16, self.first_conv_dim * 4],
                                                                      [self.first_conv_dim * 4], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            print(deconv_h2.get_shape())
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))
            # h2 = tf.nn.relu(mp.batch_norm(deconv_h2, scope_name='g_bn_h2', reuse=reuse, trainable=trainable))

            # 3rd
            deconv_h3, deconv_h3_w, deconv_h3_b = mp.conv2d_transpose('g_deconv_h3', h2,
                                                                      [5, 5, self.first_conv_dim * 2, h2.get_shape()[-1]],
                                                                      [self.batch_size, 32, 32, self.first_conv_dim * 2],
                                                                      [self.first_conv_dim * 2], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))
            # h3 = tf.nn.relu(mp.batch_norm(deconv_h3, scope_name='g_bn_h3', reuse=reuse, trainable=trainable))

            # 4th
            deconv_h4, deconv_h4_w, deconv_h4_b = mp.conv2d_transpose('g_deconv_h4', h3,
                                                                      [5, 5, 3, h3.get_shape()[-1]],
                                                                      [self.batch_size, 64, 64, 3],
                                                                      [3], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True)
            return tf.nn.tanh(deconv_h4)


    def inference(self, z, reuse=False, trainable=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("===G")
            # linear projection.
            z_, h0_w, self.h0_b = mp.linear_project('g_lin_project_h0', z, self.first_conv_dim * 8 * 4 * 4, reuse=reuse, with_w=True)

            g_fc1, w1, b1 = mp.fc_relu('g_fc1', z_, [z_.get_shape()[1], self.fc_dim], [self.fc_dim], reuse=reuse,
                                       trainable=trainable)
            # reshape for cnn inputs.
            h0 = tf.reshape(g_fc1, [-1, 4, 4, self.first_conv_dim * 8])
            # batch norm
            h0 = tf.nn.relu(self.g_bn0(h0, trainable=trainable))
            # h0 = tf.nn.relu(mp.batch_norm(h0, scope_name='g_bn_h0', reuse=reuse, trainable=trainable))

            # deconv1 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h1, deconv_h1_w, deconv_h1_b = mp.conv2d_transpose('g_deconv_h1', h0,
                                                                      [5, 5, self.first_conv_dim * 8, h0.get_shape()[-1]],
                                                                      [self.batch_size, 8, 8, self.first_conv_dim * 8],
                                                                      [self.first_conv_dim * 8], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse,  with_w=True, trainable=trainable)
            h1 = tf.nn.relu(self.g_bn1(deconv_h1, trainable=trainable))
            # h1 = tf.nn.relu(mp.batch_norm(deconv_h1, scope_name='g_bn_h1', reuse=reuse, trainable=trainable))

            # deconv2 conv2d_transpose arguments = (scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True)
            deconv_h2, deconv_h2_w, deconv_h2_b = mp.conv2d_transpose('g_deconv_h2', h1,
                                                                      [5, 5, self.first_conv_dim * 4, h1.get_shape()[-1]],
                                                                      [self.batch_size, 16, 16, self.first_conv_dim * 4],
                                                                      [self.first_conv_dim * 4], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            print(deconv_h2.get_shape())
            h2 = tf.nn.relu(self.g_bn2(deconv_h2, trainable=trainable))
            # h2 = tf.nn.relu(mp.batch_norm(deconv_h2, scope_name='g_bn_h2', reuse=reuse, trainable=trainable))

            # 3rd

            deconv_h3, deconv_h3_w, deconv_h3_b = mp.conv2d_transpose('g_deconv_h3', h2,
                                                                      [5, 5, self.first_conv_dim * 2, h2.get_shape()[-1]],
                                                                      [self.batch_size, 32, 32, self.first_conv_dim * 2],
                                                                      [self.first_conv_dim * 2], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h3 = tf.nn.relu(self.g_bn3(deconv_h3, trainable=trainable))
            # h3 = tf.nn.relu(mp.batch_norm(deconv_h3, scope_name='g_bn_h3', reuse=reuse, trainable=trainable))

            # 4th
            deconv_h4, deconv_h4_w, deconv_h4_b = mp.conv2d_transpose('g_deconv_h4', h3,
                                                                      [5, 5, 3, h3.get_shape()[-1]],
                                                                      [self.batch_size, 64, 64, 3],
                                                                      [3], [1, 2, 2, 1],
                                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            return tf.nn.tanh(deconv_h4)


class Encoder:
    # Encoder from DescriminatorExpand
    def __init__(self, batch_size, first_conv_dim, z_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.z_dim = z_dim
        self.fc_dim = 8192

        self.d_bn1 = mp.BatchNorm(name='d_bn1')
        self.d_bn2 = mp.BatchNorm(name='d_bn2')
        self.d_bn3 = mp.BatchNorm(name='d_bn3')

    def inference(self, x, reuse=False, trainable=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            print("===D")
            print(x.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('d_conv_h0', x,
                                                      [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0 = mp.lrelu(conv_h0)
            print(h0.get_shape())
            conv_h0_2, conv_h0_2_w, conv_h0_2_b = mp.conv2d('d_conv_h0_2', h0,
                                                      [5, 5, h0.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0_2 = mp.lrelu(conv_h0_2)
            print(h0_2.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('d_conv_h1', h0_2,
                                                      [5, 5, h0_2.get_shape()[-1], self.first_conv_dim*2],
                                                      [self.first_conv_dim*2],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h1 = mp.lrelu(conv_h1)
            h1 = mp.lrelu(self.d_bn1(conv_h1, trainable=trainable))
            # h1 = mp.lrelu(mp.batch_norm(conv_h1, scope_name='d_bn_h1', reuse=reuse, trainable=trainable))

            conv_h1_2, conv_h1_2_w, conv_h1_2_b = mp.conv2d('d_conv_h1_2', h1,
                                                      [5, 5, h1.get_shape()[-1], self.first_conv_dim * 2],
                                                      [self.first_conv_dim * 2],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h1_2 = mp.lrelu(conv_h1_2)
            print(h1_2.get_shape())
            # 3rd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('d_conv_h2', h1_2,
                                                      [5, 5, h1_2.get_shape()[-1], self.first_conv_dim*4],
                                                      [self.first_conv_dim*4],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h2 = mp.lrelu(conv_h2)
            h2 = mp.lrelu(self.d_bn2(conv_h2, trainable=trainable))
            # h2 = mp.lrelu(mp.batch_norm(conv_h2, scope_name='d_bn_h2', reuse=reuse, trainable=trainable))
            print(h2.get_shape())

            conv_h2_2, conv_h2_2_w, conv_h2_2_b = mp.conv2d('d_conv_h2_2', h2,
                                                      [5, 5, h2.get_shape()[-1], self.first_conv_dim * 4],
                                                      [self.first_conv_dim * 4],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            # h2 = mp.lrelu(conv_h2)
            h2_2 = mp.lrelu(conv_h2_2)
            print(h2_2.get_shape())

            # 4th
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('d_conv_h3', h2_2,
                                                      [5, 5, h2_2.get_shape()[-1], self.first_conv_dim*8],
                                                      [self.first_conv_dim*8],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h3 = mp.lrelu(conv_h3)
            h3 = mp.lrelu(self.d_bn3(conv_h3, trainable=trainable))
            # h3 = mp.lrelu(mp.batch_norm(conv_h3, scope_name='d_bn_h3', reuse=reuse, trainable=trainable))
            print("h3")
            print(h3.get_shape())

            conv_h3_2, conv_h3_2_w, conv_h3_2_b = mp.conv2d('d_conv_h3_2', h3,
                                                      [5, 5, h3.get_shape()[-1], self.first_conv_dim * 8],
                                                      [self.first_conv_dim * 8],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)

            h3_2 = mp.lrelu(conv_h3_2)
            print(h3_2.get_shape())
            e_fc1, w1, b1 = mp.fc_flat_relu('d_fc1', h3_2, [h3_2.get_shape().as_list()[1] * h3_2.get_shape().as_list()[2] * h3_2.get_shape().as_list()[3], self.fc_dim], [self.fc_dim],
                                            reuse=reuse, trainable=True)
            # linear projection (skip h3)
            h4 = mp.linear_project('e_lin_project_h4', tf.reshape(e_fc1, [self.batch_size, -1]), self.z_dim, reuse=reuse)
            return tf.nn.tanh(h4), h4, e_fc1


class DescriminatorExpand:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.fc_dim = 8192

        self.d_bn1 = mp.BatchNorm(name='d_bn1')
        self.d_bn2 = mp.BatchNorm(name='d_bn2')
        self.d_bn3 = mp.BatchNorm(name='d_bn3')

    def inference(self, x, reuse=False, trainable=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            print("===D")
            print(x.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('d_conv_h0', x,
                                                      [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0 = mp.lrelu(conv_h0)
            print(h0.get_shape())
            conv_h0_2, conv_h0_2_w, conv_h0_2_b = mp.conv2d('d_conv_h0_2', h0,
                                                      [5, 5, h0.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0_2 = mp.lrelu(conv_h0_2)
            print(h0_2.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('d_conv_h1', h0_2,
                                                      [5, 5, h0_2.get_shape()[-1], self.first_conv_dim*2],
                                                      [self.first_conv_dim*2],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h1 = mp.lrelu(conv_h1)
            h1 = mp.lrelu(self.d_bn1(conv_h1, trainable=trainable))
            # h1 = mp.lrelu(mp.batch_norm(conv_h1, scope_name='d_bn_h1', reuse=reuse, trainable=trainable))

            conv_h1_2, conv_h1_2_w, conv_h1_2_b = mp.conv2d('d_conv_h1_2', h1,
                                                      [5, 5, h1.get_shape()[-1], self.first_conv_dim * 2],
                                                      [self.first_conv_dim * 2],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h1_2 = mp.lrelu(conv_h1_2)
            print(h1_2.get_shape())
            # 3rd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('d_conv_h2', h1_2,
                                                      [5, 5, h1_2.get_shape()[-1], self.first_conv_dim*4],
                                                      [self.first_conv_dim*4],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h2 = mp.lrelu(conv_h2)
            h2 = mp.lrelu(self.d_bn2(conv_h2, trainable=trainable))
            # h2 = mp.lrelu(mp.batch_norm(conv_h2, scope_name='d_bn_h2', reuse=reuse, trainable=trainable))
            print(h2.get_shape())

            conv_h2_2, conv_h2_2_w, conv_h2_2_b = mp.conv2d('d_conv_h2_2', h2,
                                                      [5, 5, h2.get_shape()[-1], self.first_conv_dim * 4],
                                                      [self.first_conv_dim * 4],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            # h2 = mp.lrelu(conv_h2)
            h2_2 = mp.lrelu(conv_h2_2)
            print(h2_2.get_shape())

            # 4th
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('d_conv_h3', h2_2,
                                                      [5, 5, h2_2.get_shape()[-1], self.first_conv_dim*8],
                                                      [self.first_conv_dim*8],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h3 = mp.lrelu(conv_h3)
            h3 = mp.lrelu(self.d_bn3(conv_h3, trainable=trainable))
            # h3 = mp.lrelu(mp.batch_norm(conv_h3, scope_name='d_bn_h3', reuse=reuse, trainable=trainable))
            print("h3")
            print(h3.get_shape())

            conv_h3_2, conv_h3_2_w, conv_h3_2_b = mp.conv2d('d_conv_h3_2', h3,
                                                      [5, 5, h3.get_shape()[-1], self.first_conv_dim * 8],
                                                      [self.first_conv_dim * 8],
                                                      [1, 1, 1, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)

            h3_2 = mp.lrelu(conv_h3_2)
            print(h3_2.get_shape())
            e_fc1, w1, b1 = mp.fc_flat_relu('d_fc1', h3_2, [h3_2.get_shape().as_list()[1] * h3_2.get_shape().as_list()[2] * h3_2.get_shape().as_list()[3], self.fc_dim], [self.fc_dim],
                                            reuse=reuse, trainable=trainable)

            # linear projection (skip h3)
            h4 = mp.linear_project('d_lin_project_h4', tf.reshape(e_fc1, [self.batch_size, -1]), 1, reuse=reuse)
            # return tf.nn.tanh(h4), h4, e_fc1
            return tf.nn.sigmoid(h4), h4, e_fc1


class DescriminatorResnet:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim

        # self.d_bn1 = mp.BatchNorm(name='d_bn1')
        # self.d_bn2 = mp.BatchNorm(name='d_bn2')
        # self.d_bn3 = mp.BatchNorm(name='d_bn3')

    def inference(self, x, reuse=False, trainable=True, num=0):
        with tf.variable_scope("discriminator_%d" % num) as scope:
            if reuse:
                scope.reuse_variables()

            print("===D")
            print(x.get_shape())

            # TODO residual blocks

            # 1st conv
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('d_conv_h0', x,
                                                      [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0 = mp.lrelu(conv_h0)
            print(h0.get_shape())

            # 1st resblock
            # residual_block_wout_bn(scope_name, inputs, output_channel, first_block=False, wd=0.0, trainable=True):
            res_h0 = mp.residual_block_wout_bn('d_res_h0', conv_h0, self.first_conv_dim, trainable=trainable)
            res_h1 = mp.residual_block_wout_bn('d_res_h1', res_h0, self.first_conv_dim, trainable=trainable)
            res_h2 = mp.residual_block_wout_bn('d_res_h2', res_h1, self.first_conv_dim, trainable=trainable)

            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('d_conv_h1', res_h2,
                                                      [5, 5, h0.get_shape()[-1], self.first_conv_dim*2],
                                                      [self.first_conv_dim*2],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h1 = mp.lrelu(conv_h1)
            #h1 = mp.lrelu(self.d_bn1(conv_h1, trainable=trainable))
            #h1 = mp.lrelu(mp.batch_norm(conv_h1, scope_name='d_bn_h1', reuse=reuse, trainable=trainable))

            # 2nd resblock
            res_h3 = mp.residual_block_wout_bn('d_res_h3', h1, self.first_conv_dim*2, trainable=trainable)
            res_h4 = mp.residual_block_wout_bn('d_res_h4', res_h3, self.first_conv_dim*2, trainable=trainable)
            res_h5 = mp.residual_block_wout_bn('d_res_h5', res_h4, self.first_conv_dim*2, trainable=trainable)

            # 3rd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('d_conv_h2', res_h5,
                                                      [5, 5, h1.get_shape()[-1], self.first_conv_dim*4],
                                                      [self.first_conv_dim*4],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h2 = mp.lrelu(conv_h2)
            #h2 = mp.lrelu(self.d_bn2(conv_h2, trainable=trainable))
            #h2 = mp.lrelu(mp.batch_norm(conv_h2, scope_name='d_bn_h2', reuse=reuse, trainable=trainable))
            print(h2.get_shape())

            # 3rd resblockl 
            res_h6 = mp.residual_block_wout_bn('d_res_h6', h2, self.first_conv_dim * 4, trainable=trainable)
            res_h7 = mp.residual_block_wout_bn('d_res_h7', res_h6, self.first_conv_dim * 4, trainable=trainable)
            res_h8 = mp.residual_block_wout_bn('d_res_h8', res_h7, self.first_conv_dim * 4, trainable=trainable)

            # 4th
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('d_conv_h3', res_h8,
                                                      [5, 5, h2.get_shape()[-1], self.first_conv_dim*8],
                                                      [self.first_conv_dim*8],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h3 = mp.lrelu(conv_h3)
            # h3 = mp.lrelu(self.d_bn3(conv_h3, trainable=trainable))
            #h3 = mp.lrelu(mp.batch_norm(conv_h3, scope_name='d_bn_h3', reuse=reuse, trainable=trainable))
            print("h3")
            print(h3.get_shape())

            # linear projection (skip h3)
            h4 = mp.linear_project('d_lin_project_h4', tf.reshape(h3, [self.batch_size, -1]), 1, reuse=reuse)
            return tf.nn.sigmoid(h4), h4, h3


class Descriminator:
    def __init__(self, batch_size, first_conv_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim

        # self.d_bn1 = mp.BatchNorm(name='d_bn1')
        # self.d_bn2 = mp.BatchNorm(name='d_bn2')
        # self.d_bn3 = mp.BatchNorm(name='d_bn3')

    def inference(self, x, reuse=False, trainable=True, num=0):
        with tf.variable_scope("discriminator_%d" % num) as scope:
            if reuse:
                scope.reuse_variables()

            print("===D")
            print(x.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('d_conv_h0', x,
                                                      [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0 = mp.lrelu(conv_h0)
            print(h0.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('d_conv_h1', h0,
                                                      [5, 5, h0.get_shape()[-1], self.first_conv_dim*2],
                                                      [self.first_conv_dim*2],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h1 = mp.lrelu(conv_h1)
            #h1 = mp.lrelu(self.d_bn1(conv_h1, trainable=trainable))
            h1 = mp.lrelu(mp.batch_norm(conv_h1, scope_name='d_bn_h1', reuse=reuse, trainable=trainable))

            # 3rd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('d_conv_h2', h1,
                                                      [5, 5, h1.get_shape()[-1], self.first_conv_dim*4],
                                                      [self.first_conv_dim*4],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h2 = mp.lrelu(conv_h2)
            #h2 = mp.lrelu(self.d_bn2(conv_h2, trainable=trainable))
            h2 = mp.lrelu(mp.batch_norm(conv_h2, scope_name='d_bn_h2', reuse=reuse, trainable=trainable))
            print(h2.get_shape())
            # 4th
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('d_conv_h3', h2,
                                                      [5, 5, h2.get_shape()[-1], self.first_conv_dim*8],
                                                      [self.first_conv_dim*8],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h3 = mp.lrelu(conv_h3)
            # h3 = mp.lrelu(self.d_bn3(conv_h3, trainable=trainable))
            h3 = mp.lrelu(mp.batch_norm(conv_h3, scope_name='d_bn_h3', reuse=reuse, trainable=trainable))
            print("h3")
            print(h3.get_shape())

            # linear projection (skip h3)
            h4 = mp.linear_project('d_lin_project_h4', tf.reshape(h3, [self.batch_size, -1]), 1, reuse=reuse)
            return tf.nn.sigmoid(h4), h4, h3


class Reverser:
    def __init__(self, batch_size, first_conv_dim, z_dim):
        self.batch_size = batch_size
        self.first_conv_dim = first_conv_dim
        self.z_dim = z_dim

        self.d_bn1 = mp.BatchNorm(name='r_bn1')
        self.d_bn2 = mp.BatchNorm(name='r_bn2')
        self.d_bn3 = mp.BatchNorm(name='r_bn3')

    def inference(self, x, reuse=False, trainable=True):
        with tf.variable_scope("reverser") as scope:
            if reuse:
                scope.reuse_variables()

            print("===D")
            print(x.get_shape())
            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h0, conv_h0_w, conv_h0_b = mp.conv2d('r_conv_h0', x,
                                                      [5, 5, x.get_shape()[-1], self.first_conv_dim],
                                                      [self.first_conv_dim],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            h0 = mp.lrelu(conv_h0)
            print(h0.get_shape())

            # conv2d arguments = (scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False)
            conv_h1, conv_h1_w, conv_h1_b = mp.conv2d('r_conv_h1', h0,
                                                      [5, 5, h0.get_shape()[-1], self.first_conv_dim*2],
                                                      [self.first_conv_dim*2],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h1 = mp.lrelu(conv_h1)
            h1 = mp.lrelu(self.d_bn1(conv_h1, trainable=trainable))
            # h1 = mp.lrelu(mp.batch_norm(conv_h1, scope_name='d_bn_h1', reuse=reuse, trainable=trainable))

            # 3rd
            conv_h2, conv_h2_w, conv_h2_b = mp.conv2d('r_conv_h2', h1,
                                                      [5, 5, h1.get_shape()[-1], self.first_conv_dim*4],
                                                      [self.first_conv_dim*4],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h2 = mp.lrelu(conv_h2)
            h2 = mp.lrelu(self.d_bn2(conv_h2, trainable=trainable))
            # h2 = mp.lrelu(mp.batch_norm(conv_h2, scope_name='d_bn_h2', reuse=reuse, trainable=trainable))
            print(h2.get_shape())
            # 4th
            conv_h3, conv_h3_w, conv_h3_b = mp.conv2d('r_conv_h3', h2,
                                                      [5, 5, h2.get_shape()[-1], self.first_conv_dim*8],
                                                      [self.first_conv_dim*8],
                                                      [1, 2, 2, 1],
                                                      padding='SAME', reuse=reuse, with_w=True, trainable=trainable)
            #h3 = mp.lrelu(conv_h3)
            h3 = mp.lrelu(self.d_bn3(conv_h3, trainable=trainable))
            # h3 = mp.lrelu(mp.batch_norm(conv_h3, scope_name='d_bn_h3', reuse=reuse, trainable=trainable))
            print("h3")
            print(h3.get_shape())

            # linear projection (skip h3)
            h4 = mp.linear_project('r_lin_project_h4', tf.reshape(h3, [self.batch_size, -1]), self.z_dim, reuse=reuse)
            return tf.nn.tanh(h4), h4, h3


if __name__ == '__main__':
    g = Generator(10, 5)
    z = tf.placeholder(tf.float32, [None, 100], name='z')
    g.inference(z)

    d = Descriminator(10, 5)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    d.inference(x)
