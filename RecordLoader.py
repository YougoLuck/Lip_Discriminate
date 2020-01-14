import tensorflow as tf
import pickle

# ====================================Data Property====================================
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 66
IMAGE_CHANNEL = 1
NUM_CLASSES = 3
# =====================================================================================


def _parse_function(example_proto):
    features = {'feature/img': tf.FixedLenFeature((), tf.string),
                'feature/label': tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    label = tf.cast(parsed_features['feature/label'], tf.int32)
    image = tf.decode_raw(parsed_features['feature/img'], tf.uint8)
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def create_dataset(filepath, repeat, shuffle, batch_size):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size)
    return dataset



def _parse_function2(example_proto):
    features = {'feature/img': tf.FixedLenFeature((), tf.string),
                'feature/label': tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    label = tf.cast(parsed_features['feature/label'], tf.int32)
    image = tf.decode_raw(parsed_features['feature/img'], tf.uint8)
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    label = tf.one_hot(label, 2)
    return image, label


def create_dataset2(filepath, repeat, shuffle, batch_size):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function2)
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size)
    return dataset


def _parse_function3(example_proto):
    features = {'feature/img': tf.FixedLenFeature((), tf.string),
                'feature/label': tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    label = tf.cast(parsed_features['feature/label'], tf.int32)
    image = tf.decode_raw(parsed_features['feature/img'], tf.uint8)
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    label = tf.one_hot(label, 2)
    return image, label


def create_dataset3(filepath, repeat, shuffle, batch_size):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function3)
    dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size)
    return dataset
