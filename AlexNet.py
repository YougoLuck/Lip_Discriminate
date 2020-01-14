import tensorflow as tf
import RecordLoader as rl


class AlexNet:

    def __init__(self):
        self.initialize()

    def initialize(self):
        self.input_x = tf.placeholder(shape=[None, rl.IMAGE_HEIGHT, rl.IMAGE_WIDTH, rl.IMAGE_CHANNEL], dtype=tf.uint8, name='input_x')
        self.target_label = tf.placeholder(shape=[None, rl.NUM_CLASSES], dtype=tf.int32, name='target_label')
        self.channel_dropout = 0.6
        self.channel_keep_prob = tf.placeholder(dtype=tf.float32, name='channel_keep_prob')
        self.build_graph()

    def get_feed_dict(self, images, labels, isTrain=True):
        feedDict = dict()
        if isTrain:
            feedDict[self.channel_keep_prob] = 1. - self.channel_dropout
        else:
            feedDict[self.channel_keep_prob] = 1.
        feedDict[self.input_x] = images
        feedDict[self.target_label] = labels
        return feedDict

    def lrn(self, data, R, alpha, beta, name = None, bias = 1.0):
        return tf.nn.lrn(data, depth_radius=R, alpha=alpha, beta=beta, bias=bias, name=name)

    def flatten(self, data):
        [a, b, c, d] = data.shape
        ft = tf.reshape(data, [-1, b * c * d])
        return ft

    def build_graph(self):
        inputs = tf.to_float(self.input_x)
        conv1_1 = tf.layers.conv2d(inputs=inputs,
                                   filters=96,
                                   kernel_size=[5, 5],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv1_1')

        lrn_1 = self.lrn(conv1_1,R=2,alpha=2e-5,beta=0.75, name='lrn_1')
        pool1_1 = tf.layers.max_pooling2d(inputs=lrn_1,
                                          pool_size=[3, 3],
                                          strides=2,
                                          padding='VALID',
                                          name='pool1_1')
        conv2_1 = tf.layers.conv2d(inputs=pool1_1,
                                   filters=256,
                                   kernel_size=[5, 5],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv2_1')
        lrn_2 = self.lrn(conv2_1,R=2,alpha=2e-5,beta=0.75, name='lrn_2')
        pool2_1 = tf.layers.max_pooling2d(inputs=lrn_2,
                                          pool_size=[3, 3],
                                          strides=2,
                                          padding='VALID',
                                          name='pool2_1')
        conv3_1 = tf.layers.conv2d(inputs=pool2_1,
                                   filters=384,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv3_1')
        conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                                   filters=384,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv3_2')
        conv3_3 = tf.layers.conv2d(inputs=conv3_2,
                                   filters=256,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv3_3')
        pool3_1 = tf.layers.max_pooling2d(inputs=conv3_3,
                                          pool_size=[3, 3],
                                          strides=2,
                                          padding='VALID',
                                          name='pool3_1')

        ft = self.flatten(pool3_1)
        dense4_1 = tf.layers.dense(ft, 4096, name='dense4_1')
        dense4_1 = tf.nn.relu(dense4_1)
        drop = tf.nn.dropout(dense4_1, self.channel_keep_prob, name='drop_out')
        dense4_2 = tf.layers.dense(drop, 4096, name='dense4_2')
        dense4_2 = tf.nn.relu(dense4_2)
        self.logits = tf.layers.dense(dense4_2, rl.NUM_CLASSES, name='out_logits')

        self.prediction = tf.nn.softmax(self.logits, name='out_prediction')
        out_argmax = tf.argmax(self.prediction, 1, name='out_argmax')
        target_argmax = tf.argmax(self.target_label, 1, name='target_argmax')
        s0_index = tf.where(tf.equal(target_argmax, 0))
        s1_index = tf.where(tf.equal(target_argmax, 1))
        s2_index = tf.where(tf.equal(target_argmax, 2))

        self.s0_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(out_argmax, s0_index), tf.gather_nd(target_argmax, s0_index)), dtype=tf.float32), name='out_s0_accuracy')
        self.s1_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(out_argmax, s1_index), tf.gather_nd(target_argmax, s1_index)), dtype=tf.float32), name='out_s1_accuracy')
        self.s2_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(out_argmax, s2_index), tf.gather_nd(target_argmax, s2_index)), dtype=tf.float32), name='out_s2_accuracy')
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_label, logits=self.logits), name='out_cost')
        correct_prediction = tf.equal(out_argmax, target_argmax, name='out_correct_prediction')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='out_accuracy')
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)
