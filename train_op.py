from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
class Optimizer():
    def __init__(self, train_iters):
        self.train_iters = train_iters

    def momentum_optimizer(self, loss, var_list):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch,  # Current index into the dataset.
            self.train_iters // 4,  # Decay step - this decays 4 times throughout training process.
            0.95,  # Decay rate.
            staircase=True)
        # optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(loss, global_step=batch, var_list=var_list)
        return optimizer