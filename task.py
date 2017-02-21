from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import train_op
import pdist

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# define fixed parameter
TRAIN_ITERS = 10000

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

        D = discrim.mlp(input_node)
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

        lh = np.zeros(10000)
        for i in range(10000):
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



def main(_):
    pre_train(verbose=True)


if __name__ == '__main__':
    tf.app.run()



