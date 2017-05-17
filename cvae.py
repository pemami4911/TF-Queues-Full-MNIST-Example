import tensorflow as tf
import os, time
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

tf.set_random_seed(123)

dataset_path = os.path.join(os.getcwd(), "mnist")
test_labels_file = "test-labels.csv"
train_labels_file = "train-labels.csv"

BATCH_SIZE = 100
NUM_CHANNELS = 1


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

train_filepaths, train_labels = read_label_file(os.path.join(dataset_path, train_labels_file))
test_filepaths, test_labels = read_label_file(os.path.join(dataset_path, test_labels_file))

train_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
train_labels = tf.one_hot(ops.convert_to_tensor(train_labels, dtype=dtypes.int32), 10)

test_images = ops.convert_to_tensor(test_filepaths, dtype=dtypes.string)
test_labels = tf.one_hot(ops.convert_to_tensor(test_labels, dtype=dtypes.int32), 10)

train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)

# Train
train_label = train_input_queue[1]
train_file_contents = tf.read_file(train_input_queue[0])
train_images_decoded = tf.image.decode_jpeg(train_file_contents, NUM_CHANNELS)

# Test
test_label = test_input_queue[1]
test_file_contents = tf.read_file(test_input_queue[0])
test_images_decoded = tf.image.decode_jpeg(test_file_contents, NUM_CHANNELS)

train_images_decoded = tf.cast(tf.reshape(train_images_decoded, shape=[784]), dtype=tf.float32)
test_images_decoded = tf.cast(tf.reshape(test_images_decoded, shape=[784]), dtype=tf.float32)

num_preprocess_threads = 2
min_after_dequeue = 10000

train_image_batch = tf.train.shuffle_batch([train_images_decoded, train_label],
                                           batch_size=BATCH_SIZE,
                                           capacity=min_after_dequeue + 3 * BATCH_SIZE,
                                           min_after_dequeue=min_after_dequeue)

test_image_batch = tf.train.shuffle_batch([test_images_decoded, test_label],
                                          batch_size=BATCH_SIZE,
                                          capacity=min_after_dequeue + 3 * BATCH_SIZE,
                                          min_after_dequeue=min_after_dequeue)

# Softmax regression model for debugging
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

train_y = tf.matmul(train_image_batch[0], W) + b
test_y = tf.nn.softmax(tf.matmul(test_image_batch[0], W) + b)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=train_image_batch[1], logits=train_y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(test_y, 1), tf.argmax(test_image_batch[1], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create the graph
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        avg_time = 0.
        for _ in range(1000):
            start = time.time()
            sess.run([train_step])
            diff = time.time() - start
            avg_time += diff
        avg_time /= 1000.
        print("time per minibatch: {}".format(avg_time))
        test_accuracy = 0.
        for _ in range(1000):
            test_accuracy += sess.run(accuracy)
        print("test accuracy: {}".format(test_accuracy/1000.))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
