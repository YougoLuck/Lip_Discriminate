import tensorflow as tf
import RecordLoader as rl


class GoogLENet:

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

    def build_graph(self):
        inputs = tf.to_float(self.input_x)
        conv1_1 = tf.layers.conv2d(inputs=inputs,
                                   filters=64,
                                   kernel_size=[3, 3],
                                   strides=[2, 2],
                                   padding='SAME',
                                   name='conv1_1')

        conv1_1 = tf.nn.relu(conv1_1)
        pool1_1 = tf.layers.max_pooling2d(inputs=conv1_1,
                                          pool_size=[2, 2],
                                          strides=2,
                                          padding='VALID',
                                          name='pool1_1')

        conv2_1 = tf.layers.conv2d(inputs=pool1_1,
                                   filters=192,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='conv2_1')

        conv2_1 = tf.nn.relu(conv2_1)
        pool2_1 = tf.layers.max_pooling2d(inputs=conv2_1,
                                          pool_size=[2, 2],
                                          strides=2,
                                          padding='VALID',
                                          name='pool2_1')

        inception3_1 = self.inception_model(pool2_1,64,96,128,16,32,32,name='inception3_1')
        inception3_2 = self.inception_model(inception3_1,128,128,192,32,96,64,name='inception3_2')
        pool3_1 = tf.layers.max_pooling2d(inputs=inception3_2,
                                          pool_size=[2, 2],
                                          strides=2,
                                          padding='VALID',
                                          name='pool3_1')
        inception4_1 = self.inception_model(pool3_1,192,96,208,16,48,64,name='inception4_1')
        inception4_2 = self.inception_model(inception4_1,160,112,224,24,64,64,name='inception4_2')
        inception4_3 = self.inception_model(inception4_2,128,128,256,24,64,64,name='inception4_3')
        inception4_4 = self.inception_model(inception4_3,112,114,288,32,64,64,name='inception4_4')
        inception4_5 = self.inception_model(inception4_4,256,160,320,32,128,128,name='inception4_5')
        pool4_1 = tf.layers.max_pooling2d(inputs=inception4_5,
                                          pool_size=[2, 2],
                                          strides=2,
                                          padding='VALID',
                                          name='pool4_1')
        inception5_1 = self.inception_model(pool4_1,256,160,320,32,128,128,name='inception5_1')
        inception5_2 = self.inception_model(inception5_1,384,192,384,48,128,128,name='inception5_2')

        drop = tf.nn.dropout(inception5_2, self.channel_keep_prob, name='drop_out')
        ft = self.flatten(drop)
        dense7_1 = tf.layers.dense(ft, 1000, name='dense7_1')
        dense7_1 = tf.nn.relu(dense7_1)
        self.logits = tf.layers.dense(dense7_1, rl.NUM_CLASSES, name='out_logits')

        self.prediction = tf.nn.softmax(self.logits, name='out_prediction')

        out_argmax = tf.argmax(self.prediction, 1, name='out_argmax')
        target_argmax = tf.argmax(self.target_label, 1, name='target_argmax')
        s0_index = tf.where(tf.equal(target_argmax, 0))
        s1_index = tf.where(tf.equal(target_argmax, 1))
        s2_index = tf.where(tf.equal(target_argmax, 2))
        self.s1_out = tf.gather_nd(out_argmax, s0_index)
        self.s0_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.s1_out, tf.gather_nd(target_argmax, s0_index)), dtype=tf.float32), name='out_s0_accuracy')
        self.s1_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(out_argmax, s1_index), tf.gather_nd(target_argmax, s1_index)), dtype=tf.float32), name='out_s1_accuracy')
        self.s2_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather_nd(out_argmax, s2_index), tf.gather_nd(target_argmax, s2_index)), dtype=tf.float32), name='out_s2_accuracy')
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_label, logits=self.logits), name='out_cost')
        correct_prediction = tf.equal(out_argmax, target_argmax, name='out_correct_prediction')
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='out_accuracy')
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)


    def flatten(self, data):
        [a, b, c, d] = data.shape
        ft = tf.reshape(data, [-1, b * c * d])
        return ft

    def inception_model(self, data, filters_1x1_1, filters_1x1_2, filters_3x3, filters_1x1_3, filters_5x5, filters_1x1_4,name):
        conv1_1 = tf.layers.conv2d(inputs=data,
                                   filters=filters_1x1_1,
                                   kernel_size=[1, 1],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv1_1'.format(name))

        conv2_1 = tf.layers.conv2d(inputs=data,
                                   filters=filters_1x1_2,
                                   kernel_size=[1, 1],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv2_1'.format(name))

        conv2_2 = tf.layers.conv2d(inputs=conv2_1,
                                   filters=filters_3x3,
                                   kernel_size=[3, 3],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv2_2'.format(name))

        conv3_1 = tf.layers.conv2d(inputs=data,
                                   filters=filters_1x1_3,
                                   kernel_size=[1, 1],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv3_1'.format(name))

        conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                                   filters=filters_5x5,
                                   kernel_size=[5, 5],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv3_2'.format(name))

        pool1_1 = tf.layers.max_pooling2d(inputs=data,
                                          pool_size=[3, 3],
                                          strides=1,
                                          padding='SAME',
                                          name='{}/pool1_1'.format(name))

        conv4_1 = tf.layers.conv2d(inputs=pool1_1,
                                   filters=filters_1x1_4,
                                   kernel_size=[1, 1],
                                   strides=[1, 1],
                                   padding='SAME',
                                   name='{}/conv4_1'.format(name))

        inception_data = tf.concat([conv1_1, conv2_2, conv3_2, conv4_1], axis=-1)
        return inception_data
