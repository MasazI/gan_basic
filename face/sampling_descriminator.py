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
import dataset
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
flags.DEFINE_integer("image_depth_org", 3, "original image depth")
flags.DEFINE_integer("c_dim", 3, "The size of input image channel to use (will be center cropped) [3]")

flags.DEFINE_integer("num_threads", 4, "number of threads using queue")
flags.DEFINE_integer("data_type", 3, "1: hollywood, 2: lfw, 3: test")

flags.DEFINE_string("model_name", "face_h_fm", "model_name")
flags.DEFINE_string("data_dir", "data/face", "data dir path")
flags.DEFINE_string("reverser_model_name", "rface", "model_name")

flags.DEFINE_string("sample_dir", "samples", "sample_name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_float('gpu_memory_fraction', 0.3, 'gpu memory fraction.')
flags.DEFINE_string('image_path', '', 'path to image.')

flags.DEFINE_string('mode', 'batch', 'running mode. <single, batch>')

flags.DEFINE_integer("db_size", 50000, "original image width")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_float("anomaly_threshold", 0.002, "The threshold of anomaly detection.")
flags.DEFINE_string("test_dir", "data/test/00", "test data dir path")
flags.DEFINE_string("anomaly_dir", "anom", "anomaly derectory")

class DCGAN_SR():
    def __init__(self, model_name, checkpoint_dir):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def step(self, samples):

        # descriminator inference using true images
        self.discriminator = model.Descriminator(FLAGS.batch_size, FLAGS.dc_dim)
        self.D1, D1_logits, D1_inter = self.discriminator.inference(samples)

        return self.D1


def detection(image_path, verbose=False):
    # input noize to generator
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')

    # input image to reverser
    samples = tf.placeholder(
        tf.float32,
        [FLAGS.sample_num, FLAGS.image_height, FLAGS.image_width, FLAGS.c_dim],
        name='sample_inputs')

    # base model class
    dcgan = DCGAN_SR(FLAGS.model_name, FLAGS.checkpoint_dir)

    # generate vector and discriminator output
    d_logits = dcgan.step(samples)

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
            pil_img = Image.open(image[0])
            pil_img = pil_img.resize((FLAGS.image_height_org, FLAGS.image_width_org))
            img_array = np.asarray(pil_img)
            if img_array.size != FLAGS.image_height_org * FLAGS.image_width_org * FLAGS.c_dim:
                continue
            height_diff = FLAGS.image_height_org - FLAGS.image_height
            width_diff = FLAGS.image_width_org - FLAGS.image_width
            img_array = img_array[int(height_diff/2):int(height_diff/2)+FLAGS.image_height, int(width_diff/2):int(width_diff/2)+FLAGS.image_width, :]
            # input for reverser image = tf.subtract(tf.div(image, 127.5), 1.0)
            img_array = img_array / 127.5 - 1.0
            img_array = img_array[None, ...]
            d_logits_eval = sess.run([d_logits], {samples: img_array})
            if verbose:
                print("discriminator confidence:")
                print(d_logits_eval[0])
            #vectors_evals.append(vectors_eval[0])
            if d_logits_eval[0] < FLAGS.anomaly_threshold:
                print(d_logits_eval)
                print("anomaly: %s: %f" % (image[0], d_logits_eval[0]))
                fig = plt.figure()
                a = fig.add_subplot(1, 1, 1)
                lum2_img = pil_img
                imgplot = plt.imshow(lum2_img)
                a.set_title('discriminator detection')
                a.set_xlabel("confidence: %f" % d_logits_eval[0])
                out_dir = os.path.join(FLAGS.model_name, FLAGS.anomaly_dir)
                if not gfile.Exists(out_dir):
                    gfile.MakeDirs(out_dir)
                out_path = os.path.join(out_dir, "anom_%d.png" % (i))
                plt.savefig(out_path)


        if FLAGS.mode == 'sampling':
            #features_obj = features.Features(images, vectors_evals)
            pass
            # TODO save features object
        else:
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
        pil_img = pil_img.resize((64, 64))
        img_array = np.asarray(pil_img)
        #input for reverser image = tf.subtract(tf.div(image, 127.5), 1.0)
        img_array = img_array/127.5 - 1.0
        img_array = img_array[None, ...]
        d_logits_eval = sess.run([d_logits], {samples: img_array})

        print(d_logits_eval)

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
    if FLAGS.mode == "single":
        if FLAGS.image_path == "" or FLAGS.image_path is None:
            print("Please set specific image_path. --image_path <path to image or csv file witch include path>")
            return
        image_path = FLAGS.image_path
    elif FLAGS.mode == "batch":
        test_datas = dataset.Dataset(FLAGS.test_dir, FLAGS.image_height_org, FLAGS.image_width_org,
                                     FLAGS.image_depth_org, FLAGS.batch_size, FLAGS.num_threads, type=FLAGS.data_type,
                                     crop=False, filename="test.csv")
        image_path = test_datas.data_csv
    else:
        print("invalid mode.")
        return
    detection(image_path, verbose=False)


if __name__ == '__main__':
    tf.app.run()