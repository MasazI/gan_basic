from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_part


class Discriminator():
    def __init__(self, hidden_layer_dim, output_dim):
        self.hidden_layer_dim = hidden_layer_dim
        self.hidden_layer_dim2 = hidden_layer_dim - 1
        self.output_dim = output_dim

    def tiny_mlp(self, x):
        fc1, w1, b1 = model_part.fc('fc1', x, [x.get_shape()[1], self.hidden_layer_dim], [self.hidden_layer_dim])
        logits, w2, b2 = model_part.fc('fc2', fc1, [self.hidden_layer_dim, self.output_dim], [self.output_dim])
        return logits, [w1, b1, w2, b2]

    def mlp(self, x, reuse=False):
        fc1, w1, b1 = model_part.fc('fc1', x, [x.get_shape()[1], self.hidden_layer_dim], [self.hidden_layer_dim], reuse=reuse)
        fc2, w2, b2 = model_part.fc('fc2', fc1, [self.hidden_layer_dim, self.hidden_layer_dim2], [self.hidden_layer_dim2], reuse=reuse)
        logits, w3, b3 = model_part.fc('fc3', fc2, [self.hidden_layer_dim2, self.output_dim], [self.output_dim], reuse=reuse)
        return logits, [w1, b1, w2, b2, w3, b3]

    def mlp_org(self, x):
        # construct learnable parameters within local scope
        w1 = tf.get_variable("w0", [x.get_shape()[1], 6], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
        w3 = tf.get_variable("w2", [5, self.output_dim], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("b2", [self.output_dim], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
        fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
        fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
        return fc3, [w1, b1, w2, b2, w3, b3]


class Generator():
    def __init__(self, hidden_layer_dim, output_dim):
        self.hidden_layer_dim = hidden_layer_dim
        self.hidden_layer_dim2 = hidden_layer_dim - 1
        self.output_dim = output_dim

    def tiny_mlp(self, x):
        fc1, w1, b1 = model_part.fc('fc1', x, [x.get_shape()[1], self.hidden_layer_dim], [self.hidden_layer_dim])
        logits, w2, b2 = model_part.fc('fc2', fc1, [self.hidden_layer_dim, self.output_dim], [self.output_dim])
        return logits, [w1, b1, w2, b2]

    def mlp(self, x, reuse=False):
        fc1, w1, b1 = model_part.fc('fc1', x, [x.get_shape()[1], self.hidden_layer_dim], [self.hidden_layer_dim], reuse=reuse)
        fc2, w2, b2 = model_part.fc('fc2', fc1, [self.hidden_layer_dim, self.hidden_layer_dim2], [self.hidden_layer_dim2], reuse=reuse)
        logits, w3, b3 = model_part.fc('fc3', fc2, [self.hidden_layer_dim2, self.output_dim], [self.output_dim], reuse=reuse)
        return logits, [w1, b1, w2, b2, w3, b3]

    def mlp_org(self, x):
        # construct learnable parameters within local scope
        w1 = tf.get_variable("w0", [x.get_shape()[1], 6], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
        w3 = tf.get_variable("w2", [5, self.output_dim], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("b2", [self.output_dim], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
        fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
        fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
        return fc3, [w1, b1, w2, b2, w3, b3]
