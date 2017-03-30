from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
from PIL import Image
import model
import tensorflow as tf
from tensorflow.python.platform import gfile
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_integer("sample_num", 1, "The size of sample images [1]")
flags.DEFINE_integer("image_height", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("image_width", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("c_dim", 3, "The size of input image channel to use (will be center cropped) [3]")

flags.DEFINE_string("model_name", "rface", "model_name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_float('gpu_memory_fraction', 0.5, 'gpu memory fraction.')
flags.DEFINE_string('image_path', '', 'path to image.')

class DCGAN():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, samples):
        # reverser
        self.reverser = model.Reverser(FLAGS.sample_num, FLAGS.dc_dim, FLAGS.z_dim)
        self.R1, R1_logits = self.reverser.inference(samples)
        R_sum = tf.summary.histogram("R", self.R1)

        return R1_logits, R_sum


def reverse(image_path):
    samples = tf.placeholder(
        tf.float32,
        [FLAGS.sample_num, FLAGS.image_height, FLAGS.image_width, FLAGS.c_dim],
        name='sample_inputs')
    dcgan = DCGAN(FLAGS.model_name, FLAGS.checkpoint_dir)
    vectors = dcgan.step(samples)

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
        print("Model: %s" % (ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint file found")
        exit()
    print("Model restored.")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    pil_img = Image.open(image_path)
    pil_img = pil_img.resize((64, 64))
    img_array = np.asarray(pil_img)
    #input for reverser image = tf.subtract(tf.div(image, 127.5), 1.0)
    img_array = img_array/127.5 - 1.0
    img_array = img_array[None, ...]
    vectors_eval = sess.run(vectors, {samples: img_array})

    print("vector:")
    print(vectors_eval[0])

    print("finish to predict.")
    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
    print("face DCGANs Reverse.")
    if FLAGS.image_path == "" or FLAGS.image_path is None:
        print("Please set specific image_path. --image_path <path to image>")
        return
    image_path = FLAGS.image_path
    reverse(image_path)


if __name__ == '__main__':
    tf.app.run()