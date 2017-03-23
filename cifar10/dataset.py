from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from glob import glob
import tensorflow as tf
from tensorflow.python.platform import gfile
import cPickle
from PIL import Image
import numpy as np

class Dataset:
    def __init__(self, datadir, org_height, org_width, org_depth=3, batch_size=32, threads_num=4):
        self.datadir = datadir
        self._preprocess()
        self.data = glob(os.path.join(datadir, "cifar-10", "*.jpg"))
        self.train_csv = "train.csv"
        with open(self.train_csv, "w") as f:
            for image in self.data:
                f.write(image)
                f.write("\n")
        print("dataset number: %d" % (len(self.data)))
        self.data_num = len(self.data)
        self.org_height = org_height
        self.org_width = org_width
        self.org_depth = org_depth
        self.threads_num = threads_num
        self.batch_size = batch_size
        self.batch_idxs = len(self.data) / self.batch_size

    def _unpickle(self, f):
        with open(f, 'rb') as fo:
            d = cPickle.load(fo)
        return d

    def _preprocess(self):
        batches = glob(os.path.join(self.datadir, "cifar-10-batches-py", "data_batch_*"))
        output_dir = os.path.join(self.datadir, "cifar-10")
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        else:
            print("skip generate images.")
            return

        for i, batch in enumerate(batches):
            print("processing: %s" % batch)
            d = self._unpickle(batch)
            data = d["data"]
            for j, image in enumerate(data):
                output_path = os.path.join(output_dir, "%02d_%06d.jpg" % (i, j))
                img_array = image.reshape(3, 32, 32).transpose(1, 2, 0)
                img_obj = Image.fromarray(np.uint8(img_array))
                img_obj.save(output_path)

    def _generate_image_and_label_batch(self, image, min_queue_examples):
        num_preprocess_threads = self.threads_num
        images = tf.train.shuffle_batch(
            [image],
            batch_size=self.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=min_queue_examples
        )
        # Display the training images in the visualizer
        # tf.image_summary('images', images, max_images=BATCH_SIZE)
        return images

    def get_inputs(self, input_height, input_width):
        filename_queue = tf.train.string_input_producer([self.train_csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename = tf.decode_csv(serialized_example, [["path"]])
        jpeg = tf.read_file(filename[0])
        image = tf.image.decode_jpeg(jpeg, channels=3)
        #image = tf.image.resize_images(image, [input_height, input_width])
        image = tf.image.resize_images(image, [self.org_height, self.org_width])
        # resize to input size
        if not (input_height == self.org_height and input_width == self.org_width):
            image = tf.image.crop_to_bounding_box(image, offset_height=20, offset_width=20, target_height=input_height, target_width=input_width)
            image = tf.image.resize_images(image, (input_height, input_width))
        image = tf.cast(image, tf.float32)
        image = tf.subtract(tf.div(image, 127.5), 1.0)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(10000 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train. This will take a few minutes.' % min_queue_examples)
        return self._generate_image_and_label_batch(image, min_queue_examples)


if __name__ == '__main__':
    d = Dataset('data', org_height=32, org_width=32)
    d._preprocess()