import tensorflow as tf
import RecordLoader as rloader
import numpy as np
import time

MAX_EPOCH = 1
REPEAT = 1
BATCH_SIZE = 200
TEST_TFRECORD = ''
MODEL_GRAPH = ''
MODEL_PATH = ''

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(MODEL_GRAPH)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

    filename = tf.placeholder(tf.string, [None], name='filename')
    dataset = rloader.create_dataset(filename, REPEAT, 1500, BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    sess.run(iterator.initializer, feed_dict={filename: [TEST_TFRECORD]})

    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name('input_x:0')
    target = graph.get_tensor_by_name('target_label:0')
    channel_keep_prob = graph.get_tensor_by_name('channel_keep_prob:0')
    s0 = graph.get_tensor_by_name('out_s0_accuracy:0')
    s1 = graph.get_tensor_by_name('out_s1_accuracy:0')
    mean = graph.get_tensor_by_name('out_accuracy:0')

    while True:
        try:
            image_batch, label_batch = sess.run([images, labels])
            feedDict = {input: image_batch, target: label_batch, channel_keep_prob: 1.}
            s0_accuracy, s1_accuracy, accuracy = sess.run([s0, s1, mean],  feed_dict=feedDict)
            print('Test: s0_acc:{}, s1_acc:{}, mean_acc:{}'.format(s0_accuracy, s1_accuracy, accuracy))
        except tf.errors.OutOfRangeError:
            break
