from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from glob import glob
import tensorflow as tf

class Dataset:
    def __init__(self, datadir, org_height, org_width, org_depth=3, batch_size=32, threads_num=4):
        self.data = glob(os.path.join(datadir, "*.jpg"))
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
        image = tf.image.crop_to_bounding_box(image, offset_height=20, offset_width=20, target_height=input_height, target_width=input_width)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, (input_height, input_width))
        image = tf.subtract(tf.div(image, 127.5), 1.0)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(10000 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train. This will take a few minutes.' % min_queue_examples)
        return self._generate_image_and_label_batch(image, min_queue_examples)