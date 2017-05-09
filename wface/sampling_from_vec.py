from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("z_dim_v", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim_v", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim_v", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_string("model_name_v", "/media/newton/data/models/gan/wface_h_fm_gp", "model_name")
flags.DEFINE_string("data_dir_v", "/home/newton/source/gan_basic/face/data/face", "data dir path")
flags.DEFINE_string("sample_dir_v", "samples_from_reverse_z", "sample_name")
flags.DEFINE_string("checkpoint_dir_v", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("batch_size_v", 1, "The size of batch images [64]")
flags.DEFINE_float('gpu_memory_fraction_v', 0.5, 'gpu memory fraction.')
flags.DEFINE_string("sample_type_v", "type1", "sample type name.")

class DCGAN_S():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, z):
        z_sum = tf.summary.histogram("z", z)

        self.generator = model.Generator(FLAGS.batch_size_v, FLAGS.gc_dim_v)
        self.G = self.generator.inference(z)

        # descriminator inference using true images
        self.discriminator = model.Descriminator(FLAGS.batch_size_v, FLAGS.dc_dim_v)
        #self.D1, D1_logits = self.discriminator.inference(images)

        # descriminator inference using sampling with G
        self.samples = self.generator.sampler(z, reuse=True, trainable=False)
        #self.D2, D2_logits = self.discriminator.inference(self.G, reuse=True)

#        d1_sum = tf.summary.histogram("d1", self.D1)
        #d2_sum = tf.summary.histogram("d2", self.D2)
        #G_sum = tf.summary.histogram("G", self.G)

        # return images, D1_logits, D2_logits, G_sum, z_sum, d1_sum, d2_sum
        # return D2_logits, G_sum, z_sum, d1_sum, d2_sum
        #return D2_logits, G_sum, z_sum, d2_sum

    def generate_images(self, z, row=8, col=8, png=False):
        images = tf.cast(tf.multiply(tf.add(self.G, 1.0), 127.5), tf.uint8)
        print(images.get_shape())
        images = [image for image in tf.split(images, FLAGS.batch_size_v, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], axis=2))
        image = tf.concat(rows, axis=1)
        if png:
            return tf.image.encode_png(tf.squeeze(image, [0]))
        else:
            return tf.squeeze(image, [0])

def sampling(z_eval, org_image=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction_v)
    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options)) as sess:
        dcgan = DCGAN_S(FLAGS.model_name_v, FLAGS.checkpoint_dir_v)
        z = tf.placeholder(tf.float32, [None, FLAGS.z_dim_v], name='z')

        # build model
        dcgan.step(z)
        generate_images = dcgan.generate_images(z, 1, 1, png=False)

        # saver
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'g_' in var.name]
        saver = tf.train.Saver(g_vars)

        # create session
        sess.run(tf.global_variables_initializer())

        # load parameters
        model_dir = os.path.join(FLAGS.model_name_v, FLAGS.checkpoint_dir_v)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")
            exit()
        print("Model restored.")

        generated_image_eval = sess.run(generate_images, {z: z_eval})
        out_dir = os.path.join(FLAGS.model_name_v, FLAGS.sample_dir_v)
        if not gfile.Exists(out_dir):
            gfile.MakeDirs(out_dir)
        filename = os.path.join(out_dir, 'sampling_%s.png' % (FLAGS.sample_type_v))
        if org_image is None:
            with open(filename, 'wb') as f:
                f.write(generated_image_eval)
        else:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            lum_img = org_image
            imgplot = plt.imshow(lum_img)
            a.set_title('Real')

            a = fig.add_subplot(1, 2, 2)
            lum2_img = generated_image_eval
            imgplot = plt.imshow(lum2_img)
            a.set_title('Sampleing from reverse z')
            now = datetime.datetime.now()
            utime = now.strftime("%s")
            out_dir = os.path.join(FLAGS.model_name_v, FLAGS.sample_dir_v)
            if not gfile.Exists(out_dir):
                gfile.MakeDirs(out_dir)
            out_path = os.path.join(out_dir, "sampling_%s.png" % (utime))
            plt.savefig(out_path)


def main(_):
    print("face DCGANs sampling.")
    sampling()


if __name__ == '__main__':
    tf.app.run()


