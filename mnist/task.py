from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import dataset
import model
from train_op import D_train_op
from train_op import G_train_op

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("steps", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The size of sample images [64]")

flags.DEFINE_integer("y_dim", None, "dimension of dim for y")
flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_string("model_name", "mnist", "model_name")
flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_float('gpu_memory_fraction', 0.5, 'gpu memory fraction.')


class DCGAN():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, images, z):
        z_sum = tf.summary.histogram("z", z)

        self.generator = model.Generator(FLAGS.batch_size, FLAGS.gc_dim)
        G = self.generator.inference(z)

        # descriminator inference using true images
        discriminator = model.Descriminator(FLAGS.batch_size, FLAGS.dc_dim)
        self.D1, D1_logits = discriminator.inference(images)

        # descriminator inference using sampling with G
        samples = self.generator.inference(z, reuse=True)
        self.D2, D2_logits = discriminator.inference(samples, reuse=True)

        d1_sum = tf.summary.histogram("d1", self.D1)
        d2_sum = tf.summary.histogram("d2", self.D1)
        G_sum = tf.summary.histogram("G", G)

        return images, D1_logits, D2_logits, G_sum, z_sum, d1_sum, d2_sum

    def cost(self, logits, logits_):
        # real image loss (1) for descriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(self.D1)))
        # fake image loss (0) for descriminator
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_, labels=tf.zeros_like(self.D2)))
        # fake image loss (1) for generator
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_, labels=tf.ones_like(self.D2)))

        # summary
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss = d_loss_real + d_loss_fake
        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)
        return d_loss_real, d_loss_fake, d_loss_real_sum, d_loss_fake_sum, d_loss_sum, g_loss_sum, d_loss, g_loss

    def generate_images(self, z, row=8, col=8):
        images = tf.cast(tf.multiply(tf.add(self.generator.inference(z, reuse=True), 1.0), 127.5), tf.uint8)
        print(images.get_shape())
        images = [image for image in tf.split(images, FLAGS.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], axis=2))
        image = tf.concat(rows, axis=1)
        return tf.image.encode_png(tf.squeeze(image, [0]))


def train():
    mnist = dataset.load_datas()
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
    images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')

    dcgan = DCGAN(FLAGS.model_name, FLAGS.checkpoint_dir)
    images_inf, logits1, logits2, G_sum, z_sum, d1_sum, d2_sum = dcgan.step(images, z)
    d_loss_fake, d_loss_real, d_loss_real_sum, d_loss_fake_sum, d_loss_sum, g_loss_sum, d_loss, g_loss = dcgan.cost(
        logits1, logits2)

    # trainable variables
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    # train operations
    d_optim = D_train_op(d_loss, d_vars, FLAGS.learning_rate, FLAGS.beta1)
    g_optim = G_train_op(g_loss, g_vars, FLAGS.learning_rate, FLAGS.beta1)

    # sampling from z
    generate_images = dcgan.generate_images(z, 4, 4)

    # summary
    g_sum = tf.summary.merge([z_sum, d2_sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d1_sum, d_loss_real_sum, d_loss_sum])

    # initialization
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))
    writer = tf.summary.FileWriter("./logs", sess.graph_def)

    # saver
    saver = tf.train.Saver(tf.global_variables())
    # run
    sess.run(init_op)

    coord = tf.train.Coordinator()


    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    counter = 1
    start_time = time.time()

    for epoch in xrange(FLAGS.epochs):
        for idx in range(FLAGS.steps):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            batch_image = batch[0].reshape([FLAGS.batch_size, 28, 28, 1])
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)

            images_inf_eval, _, summary_str = sess.run([images_inf, d_optim, d_sum], {z: batch_z, images: batch_image})
            writer.add_summary(summary_str, counter)

            _, summary_str = sess.run([g_optim, g_sum], {z: batch_z, images: batch_image})
            writer.add_summary(summary_str, counter)

            errD_fake = sess.run(d_loss_fake, {z: batch_z, images: batch_image})
            errD_real = sess.run(d_loss_real, {z: batch_z, images: batch_image})
            errG = sess.run(g_loss, {z: batch_z, images: batch_image})
            print("epochs: %02d %04d/%04d time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
            epoch, idx, FLAGS.steps, time.time() - start_time, errD_fake + errD_real, errG))

            if np.mod(counter, 100) == 1:
                print("generate samples.")
                generated_image_eval = sess.run(generate_images, {z: batch_z})
                out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
                if not gfile.Exists(out_dir):
                    gfile.MakeDirs(out_dir)
                filename = os.path.join(out_dir, 'out_%05d.png' % counter)
                with open(filename, 'wb') as f:
                    f.write(generated_image_eval)
            counter += 1
        if np.mod(epoch, 10) == 0:
            out_dir = os.path.join(FLAGS.model_name, FLAGS.checkpoint_dir)
            if not gfile.Exists(out_dir):
                gfile.MakeDirs(out_dir)
            out_path = os.path.join(out_dir, 'model.ckpt')
            saver.save(sess, out_path, global_step=epoch)
    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
    print("mnist DCGANs.")
    train()


if __name__ == '__main__':
    tf.app.run()