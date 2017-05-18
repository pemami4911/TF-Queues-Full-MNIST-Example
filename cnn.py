"""
Some of this code is based on the tutorial at
http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/.

This is sort of an extension of that blog post to a full example of using
Tensorflow's Readers, Queues, etc., for training a CNN on MNIST

Author: Patrick Emami
"""

import os
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

dataset_path = os.path.join(os.getcwd(), "mnist")
test_labels_file = "test-labels.csv"
train_labels_file = "train-labels.csv"

BATCH_SIZE = 50
NUM_CHANNELS = 1

tf.set_random_seed(24601)


# Helper functions for pre-processing

def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        if line == "\n":
            continue
        filepath, label = line.split(",")
        filepaths.append(os.path.join(dataset_path, filepath))
        labels.append(encode_label(label))
    return filepaths, labels  # reading labels and file path


def input_pipeline(filenames):
    filepaths, labels = read_label_file(filenames)
    # convert filepaths and labels to usable format with correct dtype
    images = ops.convert_to_tensor(filepaths, dtype=dtypes.string)
    # We'll use TF's one_hot op for fast pre-processing of labels
    labels_one_hot = tf.one_hot(ops.convert_to_tensor(labels, dtype=dtypes.int32), 10)
    # This allows you to group images and labels together into a queue
    input_queue = tf.train.slice_input_producer([images, labels_one_hot], shuffle=False)
    labels_queue = input_queue[1]
    # we need to read in the images, decode them, give them a shape, and make sure they
    # have the right dtype
    images_queue = tf.cast(tf.reshape(
        tf.image.decode_jpeg(tf.read_file(input_queue[0]), NUM_CHANNELS),
        shape=[28, 28, NUM_CHANNELS]), dtype=tf.float32)
    return images_queue, labels_queue


# Code for CNN based on Deep MNIST for Experts tutorial
# https://www.tensorflow.org/get_started/mnist/pros
# Modified to use tf.get_variable :)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name=name, initializer=initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=name, initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def build_training_model(x, y):
    with tf.variable_scope('my_graph'):
        W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
        b_conv2 = bias_variable([64], 'b_conv2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
        b_fc1 = bias_variable([1024], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([1024, 10], 'W_fc2')
        b_fc2 = bias_variable([10], 'b_fc2')

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return train_step, accuracy


def build_inference_model(x, y):
    with tf.variable_scope('my_graph', reuse=True):
        W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
        b_conv2 = bias_variable([64], 'b_conv2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
        b_fc1 = bias_variable([1024], 'b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([1024, 10], 'W_fc2')
        b_fc2 = bias_variable([10], 'b_fc2')

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy


if __name__ == '__main__':
    # Normally you also want to split off a validation set, but we omit that for this example
    train_images, train_labels = input_pipeline(os.path.join(dataset_path, train_labels_file))
    test_images, test_labels = input_pipeline(os.path.join(dataset_path, test_labels_file))

    num_preprocess_threads = 3
    min_after_dequeue = 10000
    train_image_batch = tf.train.shuffle_batch([train_images, train_labels],
                                               batch_size=BATCH_SIZE,
                                               capacity=min_after_dequeue + 3 * BATCH_SIZE,
                                               min_after_dequeue=min_after_dequeue,
                                               num_threads=num_preprocess_threads)

    test_image_batch = tf.train.shuffle_batch([test_images, test_labels],
                                              batch_size=BATCH_SIZE,
                                              capacity=min_after_dequeue + 3 * BATCH_SIZE,
                                              min_after_dequeue=min_after_dequeue,
                                              num_threads=num_preprocess_threads)

    # Create two graphs for training and inference, with shared weights
    train, train_accuracy = build_training_model(train_image_batch[0], train_image_batch[1])
    inference = build_inference_model(test_image_batch[0], test_image_batch[1])

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # This is the standard "skeleton" for running code using Tensorflow queues

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(20000):
                if i % 100 == 0:
                    # The magic part - no feed_dict necessary!
                    # Every call to train_accuracy grabs the next batch
                    # automatically
                    acc = train_accuracy.eval()
                    print("step {}, training accuracy {}".format(i, acc))
                sess.run([train])

            acc = 0.
            for i in range(100):
                acc += inference.eval()
            print("test accuracy {}".format(acc / 100.))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
