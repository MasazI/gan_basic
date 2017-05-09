from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import model
from dataset import load_csv
import sampling_from_vec
import tensorflow as tf
from tensorflow.python.platform import gfile
import datetime
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gc_dim", 64, "dimension of generative filters in conv layer")
flags.DEFINE_integer("dc_dim", 64, "dimension of discriminative filters in conv layer")

flags.DEFINE_integer("sample_num", 1, "The size of sample images [1]")
flags.DEFINE_integer("image_height", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("image_width", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("image_height_org", 108, "original image height")
flags.DEFINE_integer("image_width_org", 108, "original image width")
flags.DEFINE_integer("c_dim", 3, "The size of input image channel to use (will be center cropped) [3]")

flags.DEFINE_string("model_name", "/media/newton/data/models/gan/rwface_h_fm_gp", "model_name")
flags.DEFINE_string("data_dir", "/home/newton/source/gan_basic/face/data/face", "data dir path")
#flags.DEFINE_string("reverser_model_name", "rface_h_fm_again", "model_name")

# flags.DEFINE_string("model_name", "rface", "model_name")
flags.DEFINE_string("g_model_name", "/media/newton/data/models/gan/wface_h_fm_gp", "model_name")
flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_float('gpu_memory_fraction', 0.3, 'gpu memory fraction.')
flags.DEFINE_string('image_path', '', 'path to image.')

flags.DEFINE_string('mode', 'distribution', 'running mode. <sampling, visualize>')

flags.DEFINE_integer("db_size", 50000, "original image width")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")

class DCGAN_SR():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, samples):
        # reverser
        self.reverser = model.EncoderNoBN(FLAGS.sample_num, FLAGS.dc_dim, FLAGS.z_dim)
        self.R1, R1_logits, R1_inter = self.reverser.inference(samples)

        return R1_logits


def reverse(image_path, verbose=False):
    # input noize to generator
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')

    # input image to reverser
    samples = tf.placeholder(
        tf.float32,
        [FLAGS.sample_num, FLAGS.image_height, FLAGS.image_width, FLAGS.c_dim],
        name='sample_inputs')

    # base model class
    dcgan = DCGAN_SR(FLAGS.model_name, FLAGS.checkpoint_dir)

    # generate vector
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

    path, ext = os.path.splitext(os.path.basename(image_path))
    if ext == '.csv':
        images = load_csv(image_path)
        vectors_evals = []
        for i, image in enumerate(images):
            # temporaly
            if i == FLAGS.db_size:
                break
            print("No.%d %s" % (i, image[0]))
            pil_img = Image.open(image[0], mode="r")
            pil_img = pil_img.resize((FLAGS.image_height_org, FLAGS.image_width_org))
            img_array = np.asarray(pil_img)
            if img_array.size != FLAGS.image_height_org * FLAGS.image_width_org * FLAGS.c_dim:
                continue
            # img_array = np.reshape(img_array, (FLAGS.image_height_org, FLAGS.image_width_org))
            height_diff = FLAGS.image_height_org - FLAGS.image_height
            width_diff = FLAGS.image_width_org - FLAGS.image_width
            # crop
            img_array = img_array[int(height_diff/2):int(height_diff/2)+FLAGS.image_height, int(width_diff/2):int(width_diff/2)+FLAGS.image_width, :]
            # input for reverser image = tf.subtract(tf.div(image, 127.5), 1.0)
            img_array = img_array / 127.5 - 1.0
            img_array = img_array[None, ...]
            vectors_eval = sess.run(vectors, {samples: img_array})
            if verbose:
                print(vectors_eval)
                print("vector:")
                print(vectors_eval[0])
            vectors_evals.append(vectors_eval[0])

        if FLAGS.mode == 'sampling':
            #features_obj = features.Features(images, vectors_evals)
            pass
            # TODO save features object
        elif FLAGS.mode == 'distribution':
            pass


        elif FLAGS.mode == 'visualize':
            # visualization
            print("Calculate NearestNeighbors:")
            X = np.array(vectors_evals)
            print(X.shape)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
            distances, indices = nbrs.kneighbors(X)
            print("10 ramdom samples")
            sample_index= np.random.randint(FLAGS.db_size, size=10000)
            for i, index in enumerate(sample_index):
                nbrs_sample = indices[index]
                nbrs_distance = distances[index]
                sample_relate_image = images[nbrs_sample[0]][0]
                top_1_index = nbrs_sample[1]
                top_1_nbrs_distance = nbrs_distance[1]
                if top_1_nbrs_distance >= 3.5:
                    continue

                nn_image = images[top_1_index][0]
                print("No.%d sample similarity." % i)
                print(sample_relate_image)
                print(nn_image)
                sample_relate_image_mat = mpimg.imread(sample_relate_image)

                nn_image_mat = mpimg.imread(nn_image)

                fig = plt.figure()
                a = fig.add_subplot(1, 2, 1)
                lum_img = sample_relate_image_mat
                imgplot = plt.imshow(lum_img)
                a.set_title('Sample')

                a = fig.add_subplot(1, 2, 2)
                lum2_img = nn_image_mat
                imgplot = plt.imshow(lum2_img)
                a.set_title('NearestNeighbors Top-1')
                a.set_xlabel("distance: %f" % top_1_nbrs_distance)
                now = datetime.datetime.now()
                utime = now.strftime("%s")
                out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
                if not gfile.Exists(out_dir):
                    gfile.MakeDirs(out_dir)
                out_path = os.path.join(out_dir, "%d_%s.png" % (i, utime))
                plt.savefig(out_path)

    else:
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((FLAGS.image_height_org, FLAGS.image_width_org))
        img_array = np.asarray(pil_img)
        #input for reverser image = tf.subtract(tf.div(image, 127.5), 1.0)
        # img_array = np.reshape(img_array, (FLAGS.image_height_org, FLAGS.image_width_org))
        height_diff = FLAGS.image_height_org - FLAGS.image_height
        width_diff = FLAGS.image_width_org - FLAGS.image_width
        # crop
        img_array = img_array[int(height_diff / 2):int(height_diff / 2) + FLAGS.image_height,
                    int(width_diff / 2):int(width_diff / 2) + FLAGS.image_width, :]
        org_array = img_array

        img_array = img_array/127.5 - 1.0
        img_array = img_array[None, ...]
        vectors_eval = sess.run(vectors, {samples: img_array})

        input_vector = vectors_eval[0][None, ...]
        print(input_vector)

        sampling_from_vec.sampling(input_vector, org_image=org_array)

        # regenerate_sample = sess.run(regenerate, {z: input_vector})
        # out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
        # now = datetime.datetime.now()
        # utime = now.strftime("%s")
        # if not gfile.Exists(out_dir):
        #     gfile.MakeDirs(out_dir)
        # filename = os.path.join(out_dir, "%s.png" % (utime))
        # with open(filename, 'wb') as f:
        #     f.write(regenerate_sample)

        # fig = plt.figure()
        # a = fig.add_subplot(1, 2, 1)
        # lum_img = mpimg.imread(image_path)
        # imgplot = plt.imshow(lum_img)
        # a.set_title('Original')
        #
        # a = fig.add_subplot(1, 2, 2)
        # lum2_img = regenerate_sample
        # imgplot = plt.imshow(lum2_img)
        # a.set_title('Re Sampling')
        #
        # out_dir = os.path.join(FLAGS.model_name, FLAGS.sample_dir)
        # if not gfile.Exists(out_dir):
        #     gfile.MakeDirs(out_dir)
        # now = datetime.datetime.now()
        # utime = now.strftime("%s")
        # out_path = os.path.join(out_dir, "%s.png" % (utime))
        # plt.savefig(out_path)


    print("finish to predict.")
    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
    print("face DCGANs Reverse.")
    if FLAGS.image_path == "" or FLAGS.image_path is None:
        print("Please set specific image_path. --image_path <path to image or csv file witch include path>")
        return
    image_path = FLAGS.image_path
    reverse(image_path)


if __name__ == '__main__':
    tf.app.run()