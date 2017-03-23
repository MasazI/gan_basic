#encoding: utf-8
import tensorflow as tf


def D_train_op(d_loss, d_vars, learning_rate, beta1):
    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    #d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    #d_optim = tf.train.GradientDescentOptimizer(0.0000001).minimize(d_loss, var_list=d_vars)
    return d_optim


def G_train_op(g_loss, g_vars, learning_rate, beta1):
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    return g_optim