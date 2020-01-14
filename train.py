import tensorflow as tf
from GoogLENet import GoogLENet
from AlexNet import AlexNet
import RecordLoader as rloader

MAX_EPOCH = 200
REPEAT = 1
BATCH_SIZE = 200
TRAIN_TFRECORD = ''
TEST_TFRECORD = ''
MODEL_PATH = ''

m = GoogLENet()
with tf.Session() as sess:
    filename = tf.placeholder(tf.string, [None], name='filename')
    dataset = rloader.create_dataset(filename, REPEAT, 1500, BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    test_dataset = rloader.create_dataset(filename, 10000, 1500, 200)
    test_iterator = test_dataset.make_initializable_iterator()
    test_images, test_labels = test_iterator.get_next()

    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    max_s0 = 0
    max_s1 = 0
    for epoch in range(MAX_EPOCH):
        sess.run(iterator.initializer, feed_dict={filename: [TRAIN_TFRECORD]})
        sess.run(test_iterator.initializer, feed_dict={filename: [TEST_TFRECORD]})
        step = 0
        while True:
            try:
                image_batch, label_batch = sess.run([images, labels])
                feedDict = m.get_feed_dict(image_batch, label_batch, isTrain=True)
                loss, s0_accuracy, s1_accuracy, accuracy, _ = sess.run([m.cost, m.s0_accuracy, m.s1_accuracy, m.accuracy, m.train_op],  feed_dict=feedDict)
                print('Train: epoch:{}, step:{}, loss:{}, s0_acc:{}, s1_acc:{}, mean_acc:{}'.format(epoch, step, loss, s0_accuracy, s1_accuracy, accuracy))

                test_img_batch, test_label_batch = sess.run([test_images, test_labels])
                feedDict = m.get_feed_dict(test_img_batch, test_label_batch, isTrain=False)
                s0_accuracy, s1_accuracy, accuracy = sess.run([m.s0_accuracy, m.s1_accuracy, m.accuracy],  feed_dict=feedDict)
                print('Test: epoch:{}, step:{} s0_acc:{}, s1_acc:{}, mean_acc:{}'.format(epoch, step, s0_accuracy, s1_accuracy, accuracy))
                step += 1
            except tf.errors.OutOfRangeError:
                saver.save(sess, MODEL_PATH, write_meta_graph=True, global_step=epoch)
                break
