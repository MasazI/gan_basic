from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
print(tf.__version__)
import model
import train_op
import pdist

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# define fixed parameter
TRAIN_ITERS = 1000

# define arguments
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'train', 'Mode.')
flags.DEFINE_integer('num_layers', 1, 'Number of hidden layers.')
flags.DEFINE_integer('num_units', 5, 'Number of units per hidden layer.')
flags.DEFINE_integer('output_units', 1, 'Number of units per output layer.')
flags.DEFINE_integer('mini_batch', 200, 'Number of mini-batch size.')
flags.DEFINE_float('keep_prob', 0.75, 'Keep probability for dropout.')
flags.DEFINE_float('gpu_memory_fraction', 0.2, 'gpu memory fraction.')

def pre_train(verbose=False):
    with tf.variable_scope("D_pre"):
        input_node = tf.placeholder(tf.float32, shape=(FLAGS.mini_batch, 1))
        train_labels = tf.placeholder(tf.float32, shape=(FLAGS.mini_batch, 1))

        discrim = model.Discriminator(FLAGS.num_units, FLAGS.output_units)


        D, theta = discrim.mlp(input_node)
        loss = tf.reduce_mean(tf.square(D - train_labels))

        Opt = train_op.Optimizer(TRAIN_ITERS)
        optimizer = Opt.momentum_optimizer(loss, None)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options))

        init = tf.global_variables_initializer()
        sess.run(init)

        distribution = pdist.Distribution(-1, 1)

        if verbose:
            def plot_d0(D, input_node):
                f, ax = plt.subplots(1)
                # p_data
                xs = np.linspace(-5, 5, 1000)
                ax.plot(xs, distribution.dist(xs), label='p_data')
                # decision boundary
                r = 1000  # resolution (number of points)
                xs = np.linspace(-5, 5, r)
                ds = np.zeros((r, 1))  # decision surface
                # process multiple points in parallel in a minibatch
                for i in range(int(r / FLAGS.mini_batch)):
                    x = np.reshape(xs[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)], (FLAGS.mini_batch, 1))
                    ds[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)] = sess.run(D, {input_node: x})

                ax.plot(xs, ds, label='decision boundary')
                ax.set_ylim(0, 1.1)
                plt.legend()

            # desicion boundary before pre-trained.
            plot_d0(D, input_node)
            plt.title('initial Decision Boundary')
            plt.show()

        lh = np.zeros(100)
        for i in range(100):
            # d=np.random.normal(mu,sigma,M)
            d = (np.random.random(
                FLAGS.mini_batch) - 0.5) * 10.0  # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
            labels = distribution.dist(d)
            lh[i], _ = sess.run([loss, optimizer],
                                {input_node: np.reshape(d, (FLAGS.mini_batch, 1)), train_labels: np.reshape(labels, (FLAGS.mini_batch, 1))})

        if verbose:
            # training loss
            plt.plot(lh)
            plt.title('Training Loss')
            plt.show()

            # desicion boundary after pre-trained.
            plot_d0(D, input_node)
            plt.title('pre-trained Decision Boundary')
            plt.show()

        weightsD = sess.run(theta)
        for weightD in weightsD:
            print(weightD.shape)
            print(weightD)
        sess.close()

    with tf.variable_scope("G"):
        z_node=tf.placeholder(tf.float32, shape=(FLAGS.mini_batch, 1)) # M uniform01 floats
        gen = model.Generator(FLAGS.num_units, FLAGS.output_units)
        G,theta_g = gen.mlp(z_node) # generate normal transformation of Z
        G = tf.multiply(5.0, G)

    with tf.variable_scope("D") as scope:
        # D(x)
        x_node=tf.placeholder(tf.float32, shape=(FLAGS.mini_batch,1)) # input M normally distributed floats

        discrim = model.Discriminator(FLAGS.num_units, FLAGS.output_units)
        fc, theta_d = discrim.mlp(x_node) # output likelihood of being normally distributed

        D1 = tf.maximum(tf.minimum(fc, .99), 0.01) # clamp as a probability
        # make a copy of D that uses the same variables, but takes in G as input
        scope.reuse_variables()
        fc, theta_d = discrim.mlp(G)
        D2 = tf.maximum(tf.minimum(fc, .99), 0.01)

    obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
    obj_g=tf.reduce_mean(tf.log(D2))

    Opt = train_op.Optimizer(TRAIN_ITERS)

    opt_d = Opt.momentum_optimizer(1 - obj_d, theta_d)
    opt_g = Opt.momentum_optimizer(1 - obj_g, theta_g)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    for i, v in enumerate(theta_d):
        sess.run(v.assign(weightsD[i]))

    if verbose:
        def plot_fig():
            # plots pg, pdata, decision boundary
            f, ax = plt.subplots(1)
            # p_data
            xs = np.linspace(-5, 5, 1000)
            ax.plot(xs, distribution.dist(xs), label='p_data')

            # decision boundary
            r = 5000  # resolution (number of points)
            xs = np.linspace(-5, 5, r)
            ds = np.zeros((r, 1))  # decision surface
            # process multiple points in parallel in same minibatch
            for i in range(int(r / FLAGS.mini_batch)):
                x = np.reshape(xs[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)], (FLAGS.mini_batch, 1))
                ds[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)] = sess.run(D1, {x_node: x})

            ax.plot(xs, ds, label='decision boundary')

            # distribution of inverse-mapped points
            zs = np.linspace(-5, 5, r)
            gs = np.zeros((r, 1))  # generator function
            for i in range(int(r / FLAGS.mini_batch)):
                z = np.reshape(zs[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)], (FLAGS.mini_batch, 1))
                gs[FLAGS.mini_batch * i:FLAGS.mini_batch * (i + 1)] = sess.run(G, {z_node: z})
            histc, edges = np.histogram(gs, bins=10)
            ax.plot(np.linspace(-5, 5, 10), histc / float(r), label='p_g')

            # ylim, legend
            ax.set_ylim(0, 1.1)
            plt.legend()

        plot_fig()
        plt.title('Before Training')
        plt.show()

    # Algorithm 1 of Goodfellow et al 2014
    k = 1
    histd, histg = np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)
    for i in range(TRAIN_ITERS):
        for j in range(k):
            x = np.random.normal(-1, 1, FLAGS.mini_batch)  # sampled m-batch from p_data
            x.sort()
            z = np.linspace(-5.0, 5.0, FLAGS.mini_batch) + np.random.random(FLAGS.mini_batch) * 0.01  # sample m-batch from noise prior
            # train discriminator
            histd[i], _ = sess.run([obj_d, opt_d], {x_node: np.reshape(x, (FLAGS.mini_batch, 1)), z_node: np.reshape(z, (FLAGS.mini_batch, 1))})
        z = np.linspace(-5.0, 5.0, FLAGS.mini_batch) + np.random.random(FLAGS.mini_batch) * 0.01  # sample noise prior
        # train generator
        histg[i], _ = sess.run([obj_g, opt_g], {z_node: np.reshape(z, (FLAGS.mini_batch, 1))})  # update generator
        if i % (TRAIN_ITERS // 10) == 0:
            print(float(i) / float(TRAIN_ITERS))

    plt.plot(range(TRAIN_ITERS), histd, label='obj_d')
    plt.plot(range(TRAIN_ITERS), 1 - histg, label='obj_g')
    plt.legend()
    plt.show()
    plot_fig()
    plt.show()


def main(_):
    pre_train(verbose=True)


if __name__ == '__main__':
    tf.app.run()



