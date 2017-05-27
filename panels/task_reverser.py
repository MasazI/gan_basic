from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import dataset
import model
from train_op import D_train_op
from train_op import G_train_op
from train_op import R_train_op

from scipy import spatial
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 101, "Epoch to train [25]")
flags.DEFINE_integer("steps", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The size of sample images [64]")
flags.DEFINE_integer("image_height", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("image_width", 64, "The size of image to use (will be center cropped) [64]")

flags.DEFINE_integer("image_height_org", 108, "original image height")
flags.DEFINE_integer("image_width_org", 108, "original image width")
flags.DEFINE_integer("image_depth_org", 3, "original image depth")
flags.DEFINE_integer("num_threads", 4, "number of threads using queue")

flags.DEFINE_integer("y_dim", None, "dimension of dim for y")
flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_string("model_name", "/media/newton/data/models/panels", "model_name")
flags.DEFINE_string("data_dir", "/media/newton/data/images/panels/panel/DJI_0014", "data dir path")
flags.DEFINE_string("reverser_model_name", "/media/newton/data/models/rpanels_l4", "model_name")

flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("logs_dir", "logs", "logs dir name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_float('gpu_memory_fraction', 0.2, 'gpu memory fraction.')

flags.DEFINE_integer("data_type", 1, "1: hollywood, 2: lfw")
flags.DEFINE_bool("is_crop", False, "crop training images?")
flags.DEFINE_float('l2distace', 0.5, 'l2d rate.')
flags.DEFINE_float('clip_coss_loss', 2.0, 'loss value for clipping.')

class DCGAN():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, z):
        z_sum = tf.summary.histogram("z", z)

        # generater
        self.generator = model.Generator(FLAGS.batch_size, FLAGS.gc_dim)
        # self.G = self.generator.inference(z)

        # sampler using generator
        self.samples = self.generator.sampler(z, reuse=False, trainable=False)

        # reverser
        self.reverser = model.Encoder(FLAGS.batch_size, FLAGS.dc_dim, FLAGS.z_dim)
        self.R1, R1_logits, R1_inter = self.reverser.inference(self.samples)
        R_sum = tf.summary.histogram("R", self.R1)
                # return images, D1_logits, D2_logits, G_sum, z_sum, d1_sum, d2_sum
        # return D2_logits, G_sum, z_sum, d1_sum, d2_sum
        return R1_logits, R1_inter, R_sum, z_sum

    def cost(self, R1_logits, z_noise):
        # loss
        r_loss1 = tf.reduce_mean(tf.square(R1_logits - z_noise))
        print("*"*100)
        normed_R1_losgits = tf.nn.l2_normalize(R1_logits, dim=1)
        normed_z_noise = tf.nn.l2_normalize(z_noise, dim=1)
        r_loss2 = tf.reduce_mean(tf.losses.cosine_distance(normed_R1_losgits, normed_z_noise, dim=1))
        r_loss2 = tf.clip_by_value(r_loss2, -FLAGS.clip_coss_loss, FLAGS.clip_coss_loss)
        print(r_loss2.shape)
        r1_abs = tf.matmul(R1_logits, R1_logits, transpose_b=True)
        z_abs = tf.matmul(z_noise, z_noise, transpose_b=True)
        r_loss3 = tf.reduce_mean(tf.abs(tf.subtract(r1_abs, z_abs)))

        #         normed_embedding = tf.nn.l2_normalize(R1_logits, dim=1)
#         normed_array = tf.nn.l2_normalize(z_noise, dim=1)
#
#         cosine_similarity = tf.reduce_mean(tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0])))
# #        r_loss2 = tf.div(1.0, cosine_similarity)
#         r_loss2 = cosine_similarity

        # r1, r2 summary
        r_loss1_sum = tf.summary.scalar("r_loss1", r_loss1)
        r_loss2_sum = tf.summary.scalar("r_loss2", r_loss2)
        r_loss3_sum = tf.summary.scalar("r_loss3", r_loss3)

        # rloss
        # r_loss = tf.multiply(r_loss1, FLAGS.l2distace) + r_loss2
        # r_loss = r_loss1 + r_loss2 + r_loss3
        r_loss = r_loss1 + r_loss2
        # r summary
        r_loss_sum = tf.summary.scalar("r_loss", r_loss)

        return r_loss, r_loss_sum, r_loss1_sum, r_loss2_sum, r_loss3_sum

    def generate_images(self, z, row=8, col=8):
        images = tf.cast(tf.multiply(tf.add(self.samples, 1.0), 127.5), tf.uint8)
        print(images.get_shape())
        images = [image for image in tf.split(images, FLAGS.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], axis=2))
        image = tf.concat(rows, axis=1)
        return tf.image.encode_png(tf.squeeze(image, [0]))


def train():
    if FLAGS.data_type == 1:
        # datadir, org_height, org_width, org_depth=3, batch_size=32, threads_num=4
        datas = dataset.Dataset(FLAGS.data_dir, FLAGS.image_height_org, FLAGS.image_width_org,
                            FLAGS.image_depth_org, FLAGS.batch_size, FLAGS.num_threads, type=FLAGS.data_type, crop=True)
    elif FLAGS.data_type == 2:
        datas = dataset.Dataset(FLAGS.data_dir, FLAGS.image_height_org, FLAGS.image_width_org,
                            FLAGS.image_depth_org, FLAGS.batch_size, FLAGS.num_threads, type=FLAGS.data_type, crop=False)
    else:
        print("invalid data type.")
        return
    # images = datas.get_inputs(FLAGS.image_height, FLAGS.image_width)

    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')

    dcgan = DCGAN(FLAGS.model_name, FLAGS.checkpoint_dir)
    R1_logits, R1_inter, R_sum, z_sum = dcgan.step(z)
    r_loss, r_loss_sum, r_loss1_sum, r_loss2_sum, r_loss3_sum = dcgan.cost(R1_logits, z)

    # trainable variables
    t_vars = tf.trainable_variables()
    #g_vars = [var for var in t_vars if 'g_' in var.name]
    r_vars = [var for var in t_vars if ('e_' in var.name) or ('d_fc1' in var.name)]
    # train operations
    r_optim = R_train_op(r_loss, r_vars, FLAGS.learning_rate, FLAGS.beta1)

    all_vars = tf.global_variables()
    dg_vars = [var for var in all_vars if ('d_' in var.name) or ('g_' in var.name)]
    # saver of d and g
    saver = tf.train.Saver(dg_vars)

    # saver of e_
    saver_e = tf.train.Saver(r_vars)

    # saver of all variables
    saver_all = tf.train.Saver()

    # initialization
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))
    log_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.logs_dir)
    if gfile.Exists(log_dir):
        gfile.DeleteRecursively(log_dir)
    writer = tf.summary.FileWriter(log_dir, sess.graph_def)

    # run
    sess.run(init_op)

    # load parameters
    print("G and D Model.")
    model_dir = os.path.join(FLAGS.model_name, FLAGS.checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model: %s" % (ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("G and D Model: No checkpoint file found")
        exit()
    print("G and D Model: restored.")

    # load e parameters
    print("E Model. %s" % (FLAGS.reverser_model_name))
    model_e_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
    ckpt_e = tf.train.get_checkpoint_state(model_e_dir)
    if ckpt_e and ckpt_e.model_checkpoint_path:
        print("Model: %s" % (ckpt_e.model_checkpoint_path))
        saver_e.restore(sess, ckpt_e.model_checkpoint_path)
        print("E Model: restored.")
    else:
        print("E model: No checkpoint file found")

    # summary
    r_sum = tf.summary.merge([z_sum, R_sum, r_loss_sum, r_loss1_sum, r_loss2_sum, r_loss3_sum])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    counter = 1
    start_time = time.time()

    for epoch in xrange(FLAGS.epochs):
        for idx in xrange(0, int(datas.batch_idxs)):
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)

            # R optimization
            _, summary_str = sess.run([r_optim, r_sum], {z: batch_z})
            writer.add_summary(summary_str, counter)

            errR = sess.run(r_loss, {z: batch_z})
            print("epochs: %02d %04d/%04d time: %4.4f, r_loss: %.8f" % (
                epoch, idx, FLAGS.steps, time.time() - start_time, errR))

            # if np.mod(counter, 100) == 1:
            #     print("generate samples.")
            #     generated_image_eval = sess.run(generate_images, {z: batch_z})
            #     out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
            #     if not gfile.Exists(out_dir):
            #         gfile.MakeDirs(out_dir)
            #     filename = os.path.join(out_dir, 'out_%05d.png' % counter)
            #     with open(filename, 'wb') as f:
            #         f.write(generated_image_eval)
            counter += 1
        if np.mod(epoch, 1) == 0:
            out_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
            if not gfile.Exists(out_dir):
                gfile.MakeDirs(out_dir)
            out_path = os.path.join(out_dir, 'model.ckpt')
            saver_all.save(sess, out_path, global_step=epoch)
    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
    print("face DCGANs Reverse.")
    train()


if __name__ == '__main__':
    tf.app.run()