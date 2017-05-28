from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from PIL import Image
import dataset
import model
from train_op import D_train_op
from train_op import G_train_op
from train_op import R_train_op

from scipy import spatial
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import gfile
from glob import glob
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 101, "Epoch to train [25]")
flags.DEFINE_integer("steps", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 1, "The size of sample images [64]")
flags.DEFINE_integer("image_height", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("image_width", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("c_dim", 3, "The size of input image channel to use (will be center cropped) [3]")

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
flags.DEFINE_string("reverser_model_name", "/media/newton/data/models/rpanels_image2", "model_name")
flags.DEFINE_string("ano_dir", "samples_from_reverse", "ano dir name")

flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("logs_dir", "logs", "logs dir name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_float('gpu_memory_fraction', 0.2, 'gpu memory fraction.')

flags.DEFINE_string("mode", "test", "<query> or <test>")
flags.DEFINE_string('image_path', '', 'path to image.')
flags.DEFINE_string('test_dir', '', 'path to image.')
flags.DEFINE_integer("test_num", 5000, "umber of images for test")

flags.DEFINE_integer("data_type", 1, "1: hollywood, 2: lfw")
flags.DEFINE_bool("is_crop", False, "crop training images?")
flags.DEFINE_float('l2distace', 0.5, 'l2d rate.')
flags.DEFINE_float('clip_coss_loss', 2.0, 'loss value for clipping.')

class DCGAN():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, images):
        I_sum = tf.summary.image("images", images)

        # reverser
        self.reverser = model.Encoder(FLAGS.batch_size, FLAGS.dc_dim, FLAGS.z_dim)
        self.R1, R1_logits, R1_inter = self.reverser.inference(images)

        # generater
        self.generator = model.Generator(FLAGS.batch_size, FLAGS.gc_dim)
        # self.G = self.generator.inference(z)

        # sampler using generator
        self.samples = self.generator.sampler(self.R1, reuse=False, trainable=False)
        G_sum = tf.summary.image("generate", self.samples)

        R_sum = tf.summary.histogram("R", self.R1)
        # return images, D1_logits, D2_logits, G_sum, z_sum, d1_sum, d2_sum
        # return D2_logits, G_sum, z_sum, d1_sum, d2_sum
        return self.samples, R1_inter, R_sum, G_sum, I_sum

    def cost(self, samples, images):
        # loss
        r_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(samples, images))
        # print("*"*100)
        # normed_R1_losgits = tf.nn.l2_normalize(R1_logits, dim=1)
        # normed_z_noise = tf.nn.l2_normalize(z_noise, dim=1)
        # r_loss2 = tf.reduce_mean(tf.losses.cosine_distance(normed_R1_losgits, normed_z_noise, dim=1))
        # r_loss2 = tf.clip_by_value(r_loss2, -FLAGS.clip_coss_loss, FLAGS.clip_coss_loss)
        # print(r_loss2.shape)
        # r1_abs = tf.matmul(R1_logits, R1_logits, transpose_b=True)
        # z_abs = tf.matmul(z_noise, z_noise, transpose_b=True)
        # r_loss3 = tf.reduce_mean(tf.abs(tf.subtract(r1_abs, z_abs)))

        #         normed_embedding = tf.nn.l2_normalize(R1_logits, dim=1)
#         normed_array = tf.nn.l2_normalize(z_noise, dim=1)
#
#         cosine_similarity = tf.reduce_mean(tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0])))
# #        r_loss2 = tf.div(1.0, cosine_similarity)
#         r_loss2 = cosine_similarity

        # r1, r2 summary
        r_loss1_sum = tf.summary.scalar("r_loss1", r_loss1)
        # r_loss2_sum = tf.summary.scalar("r_loss2", r_loss2)
        # r_loss3_sum = tf.summary.scalar("r_loss3", r_loss3)

        # rloss
        # r_loss = tf.multiply(r_loss1, FLAGS.l2distace) + r_loss2
        # r_loss = r_loss1 + r_loss2 + r_loss3
        r_loss = r_loss1
        # r summary
        r_loss_sum = tf.summary.scalar("r_loss", r_loss)

        return r_loss, r_loss_sum, r_loss1_sum

    def generate_images(self, z, row=8, col=8):
        images = tf.cast(tf.multiply(tf.add(self.samples, 1.0), 127.5), tf.uint8)
        print(images.get_shape())
        images = [image for image in tf.split(images, FLAGS.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], axis=2))
        image = tf.concat(rows, axis=1)
        return tf.image.encode_png(tf.squeeze(image, [0]))


def test_all(test_dir):
    datas = glob(os.path.join(test_dir, "*.jpg"))

    query = tf.placeholder(
        tf.float32,
        [FLAGS.sample_size, FLAGS.image_height, FLAGS.image_width, FLAGS.c_dim],
        name='sample_inputs')

    dcgan = DCGAN(FLAGS.model_name, FLAGS.checkpoint_dir)
    samples, R1_inter, R_sum, G_sum, I_sum = dcgan.step(query)

    # # trainable variables
    # t_vars = tf.trainable_variables()
    # # g_vars = [var for var in t_vars if 'g_' in var.name]
    # r_vars = [var for var in t_vars if ('e_' in var.name) or ('d_fc1' in var.name)]
    #
    # all_vars = tf.global_variables()
    # dg_vars = [var for var in all_vars if ('d_' in var.name) or ('g_' in var.name)]
    # # saver of d and g
    # saver = tf.train.Saver(dg_vars)
    #
    # # saver of e_
    # saver_e = tf.train.Saver(r_vars)

    # saver of all variables
    saver_all = tf.train.Saver()

    # initialization
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))

    # run
    sess.run(init_op)

    # load parameters
    print("G, E and D Model.")
    model_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model: %s" % (ckpt.model_checkpoint_path))
        saver_all.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("G, E and D Model: No checkpoint file found")
        exit()
    print("G, E and D Model: restored.")

    # # load e parameters
    # print("E Model. %s" % (FLAGS.reverser_model_name))
    # model_e_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
    # ckpt_e = tf.train.get_checkpoint_state(model_e_dir)
    # if ckpt_e and ckpt_e.model_checkpoint_path:
    #     print("Model: %s" % (ckpt_e.model_checkpoint_path))
    #     saver_e.restore(sess, ckpt_e.model_checkpoint_path)
    #     print("E Model: restored.")
    # else:
    #     print("E model: No checkpoint file found")

    now = datetime.datetime.now()
    utime = now.strftime("%s")

    for i, image_path in enumerate(datas):
        if i >= FLAGS.test_num:
            return
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((64, 64))
        img_array = np.asarray(pil_img)
        org_array = img_array

        img_array = img_array / 127.5 - 1.0
        img_array = img_array[None, ...]

        samples_val = sess.run(samples, {query: img_array})
        print(samples_val)
        samples_val = (samples_val + 1.0) * 127.5

        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        lum_img = org_array
        imgplot = plt.imshow(lum_img)
        a.set_title('Real')

        a = fig.add_subplot(1, 2, 2)
        lum2_img = np.uint8(samples_val[0])
        imgplot = plt.imshow(lum2_img)
        a.set_title('Sampleing from reverse z')

        out_dir = os.path.join(FLAGS.model_name, FLAGS.ano_dir, utime)
        if not gfile.Exists(out_dir):
            gfile.MakeDirs(out_dir)
        out_path = os.path.join(out_dir, "sampling_%d.png" % (i))
        plt.savefig(out_path)

    sess.close()


def test(image_path):
    query = tf.placeholder(
        tf.float32,
        [FLAGS.sample_size, FLAGS.image_height, FLAGS.image_width, FLAGS.c_dim],
        name='sample_inputs')

    dcgan = DCGAN(FLAGS.model_name, FLAGS.checkpoint_dir)
    samples, R1_inter, R_sum, G_sum, I_sum = dcgan.step(query)

    # # trainable variables
    # t_vars = tf.trainable_variables()
    # # g_vars = [var for var in t_vars if 'g_' in var.name]
    # r_vars = [var for var in t_vars if ('e_' in var.name) or ('d_fc1' in var.name)]
    #
    # all_vars = tf.global_variables()
    # dg_vars = [var for var in all_vars if ('d_' in var.name) or ('g_' in var.name)]
    # # saver of d and g
    # saver = tf.train.Saver(dg_vars)
    #
    # # saver of e_
    # saver_e = tf.train.Saver(r_vars)

    # saver of all variables
    saver_all = tf.train.Saver()

    # initialization
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))

    # run
    sess.run(init_op)

    # load parameters
    print("G, E and D Model.")
    model_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model: %s" % (ckpt.model_checkpoint_path))
        saver_all.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("G, E and D Model: No checkpoint file found")
        exit()
    print("G, E and D Model: restored.")

    # # load e parameters
    # print("E Model. %s" % (FLAGS.reverser_model_name))
    # model_e_dir = os.path.join(FLAGS.reverser_model_name, FLAGS.checkpoint_dir)
    # ckpt_e = tf.train.get_checkpoint_state(model_e_dir)
    # if ckpt_e and ckpt_e.model_checkpoint_path:
    #     print("Model: %s" % (ckpt_e.model_checkpoint_path))
    #     saver_e.restore(sess, ckpt_e.model_checkpoint_path)
    #     print("E Model: restored.")
    # else:
    #     print("E model: No checkpoint file found")

    pil_img = Image.open(image_path)
    pil_img = pil_img.resize((64, 64))
    img_array = np.asarray(pil_img)
    org_array = img_array

    img_array = img_array / 127.5 - 1.0
    img_array = img_array[None, ...]


    samples_val = sess.run(samples, {query: img_array})
    print(samples_val)
    samples_val = (samples_val + 1.0) * 127.5

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    lum_img = org_array
    imgplot = plt.imshow(lum_img)
    a.set_title('Real')

    a = fig.add_subplot(1, 2, 2)
    lum2_img = np.uint8(samples_val[0])
    imgplot = plt.imshow(lum2_img)
    a.set_title('Sampleing from reverse z')
    now = datetime.datetime.now()
    utime = now.strftime("%s")
    out_dir = os.path.join(FLAGS.model_name, FLAGS.ano_dir)
    if not gfile.Exists(out_dir):
        gfile.MakeDirs(out_dir)
    out_path = os.path.join(out_dir, "sampling_%s.png" % (utime))
    plt.savefig(out_path)

    sess.close()


def main(_):
    if FLAGS.mode == 'query':
        print("panels DCGANs anomaly sampling.")
        if FLAGS.image_path == "" or FLAGS.image_path is None:
            print("Please set specific image_path. --image_path <path to image or csv file witch include path>")
            return
        image_path = FLAGS.image_path
        test(image_path)
    elif FLAGS.mode == 'test':
        print("panels DCGANs anomaly sampling test.")
        if FLAGS.test_dir == "" or FLAGS.test_dir is None:
            print("Please set specific test_dir. --test_dir <path to test_dir including test images>")
            return
        test_dir = FLAGS.test_dir
        test_all(test_dir)


if __name__ == '__main__':
    tf.app.run()