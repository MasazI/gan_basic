from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import model
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_string("model_name", "face_h_fm_ex", "model_name")
flags.DEFINE_string("data_dir", "data/face", "data dir path")
flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("batch_size", 144, "The size of batch images [64]")
flags.DEFINE_float('gpu_memory_fraction', 0.5, 'gpu memory fraction.')
flags.DEFINE_string("sample_type", "type1", "sample type name.")

class DCGAN_S():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, z):
        z_sum = tf.summary.histogram("z", z)

        self.generator = model.Generator(FLAGS.batch_size, FLAGS.gc_dim)
        self.G = self.generator.inference(z)

        # descriminator inference using true images
        self.discriminator = model.DescriminatorExpand(FLAGS.batch_size, FLAGS.dc_dim)
        #self.D1, D1_logits = self.discriminator.inference(images)

        # descriminator inference using sampling with G
        self.samples = self.generator.sampler(z, reuse=True)
        self.D2, D2_logits, D2_inter = self.discriminator.inference(self.G, reuse=False)

#        d1_sum = tf.summary.histogram("d1", self.D1)
        d2_sum = tf.summary.histogram("d2", self.D2)
        G_sum = tf.summary.histogram("G", self.G)

        # return images, D1_logits, D2_logits, G_sum, z_sum, d1_sum, d2_sum
        # return D2_logits, G_sum, z_sum, d1_sum, d2_sum
        return D2_logits, G_sum, z_sum, d2_sum

    def generate_images(self, z, row=8, col=8):
        images = tf.cast(tf.multiply(tf.add(self.samples, 1.0), 127.5), tf.uint8)
        print(images.get_shape())
        images = [image for image in tf.split(images, FLAGS.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], axis=2))
        image = tf.concat(rows, axis=1)
        return tf.image.encode_png(tf.squeeze(image, [0]))

def sampling():
    dcgan = DCGAN_S(FLAGS.model_name, FLAGS.checkpoint_dir)
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')

    # build model
    dcgan.step(z)
    generate_images = dcgan.generate_images(z, 12, 12)

    # saver
    saver = tf.train.Saver()

    # create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())

    # load parameters
    model_dir = os.path.join(FLAGS.model_name, FLAGS.checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint file found")
        exit()
    print("Model restored.")

    batch_z = np.zeros([FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
    initial_value = 0
    for j in xrange(100):
        for i in xrange(FLAGS.batch_size):
           batch_z[i, j] = 1.0 - i*2.0/400.

        generated_image_eval = sess.run(generate_images, {z: batch_z})
        out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
        if not gfile.Exists(out_dir):
            gfile.MakeDirs(out_dir)
        filename = os.path.join(out_dir, 'sampling_%s_dim%d.png' % (FLAGS.sample_type, j))
        with open(filename, 'wb') as f:
            f.write(generated_image_eval)

def main(_):
    print("face DCGANs sampling.")
    sampling()


if __name__ == '__main__':
    tf.app.run()


