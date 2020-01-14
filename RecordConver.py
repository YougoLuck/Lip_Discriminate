import os
import tensorflow as tf
import RecordUtil as rutil
import pickle
import cv2
import os
import numpy as np
from random import shuffle,randint,uniform


def conver2Record(dataset_path, output, data_each_speaker_cnt):
    dataset_subjects = ['lip', 'face', 'background']
    img_paths = []
    labels = []
    for dataset_subject in dataset_subjects:
        subject_path = os.path.join(dataset_path, dataset_subject)
        img_labels = []
        if dataset_subject == 'background':
            img_names = os.listdir(subject_path)
            img_names = img_names[:int(data_each_speaker_cnt*33 / 2)]
            tem_paths = []
            for img_name in img_names:
                tem_paths.append(os.path.join(subject_path, img_name))
                # tem_paths.append(os.path.join(subject_path, '{}+{}'.format(img_name, 'roate')))
                # tem_paths.append(os.path.join(subject_path, '{}+{}'.format(img_name, 'move')))
            img_labels = [2] * len(tem_paths)
            img_paths = img_paths + tem_paths

        else:
            speakers = os.listdir(subject_path)
            for speaker in speakers:
                if dataset_subject == 'lip':
                    tem_cnt = data_each_speaker_cnt
                else:
                    tem_cnt = int(data_each_speaker_cnt / 2)
                speaker_path = os.path.join(subject_path, speaker)
                img_names = os.listdir(speaker_path)[:tem_cnt]
                tem_paths = []
                for img_name in img_names:
                    tem_paths.append(os.path.join(speaker_path, img_name))
                    # tem_paths.append(os.path.join(speaker_path, '{}+{}'.format(img_name, 'roate')))
                    # tem_paths.append(os.path.join(speaker_path, '{}+{}'.format(img_name, 'move')))

                if dataset_subject == 'lip':
                    img_labels = img_labels + ([0] * len(tem_paths))
                else:
                    img_labels = img_labels + ([1] * len(tem_paths))

                img_paths = img_paths + tem_paths
        labels = labels + img_labels
    index_shuf = list(range(len(img_paths)))
    shuffle(index_shuf)
    shuffle_imgPaths = list()
    shuffle_labels = list()
    for i in index_shuf:
        shuffle_imgPaths.append(img_paths[i])
        shuffle_labels.append(labels[i])

    output_dir = os.path.dirname(output)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    with tf.python_io.TFRecordWriter(output) as tfWriter:
        tem = [-1, 1]
        for index in range(len(shuffle_imgPaths)):
            file = shuffle_imgPaths[index]
            out_split = file.split('+')
            file = out_split[0]
            if len(out_split) == 2:
                op_flag = output[1]
            else:
                op_flag = None
            label = shuffle_labels[index]
            img = cv2.imread(file)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rows,cols = img.shape[:2]
            if op_flag == 'roate':
                roate_angle = randint(20, 40) * tem[randint(0, 1)]
                M_roate = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), roate_angle, 1+uniform(0, 0.2))
                img = cv2.warpAffine(img, M_roate, (cols,rows))
            elif op_flag == 'move':
                move_x = randint(10, 20) * tem[randint(0, 1)]
                move_y = randint(10, 20) * tem[randint(0, 1)]
                M_move = np.float32([[1,0,move_x],[0,1,move_y]])
                img = cv2.warpAffine(img, M_move, (cols,rows))
            print('index:{}, file:{}, label:{}, shape:{}'.format(index, file, label, img.shape))
            example = conver2Example(img, label)
            tfWriter.write(example.SerializeToString())


def conver2Example(img, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'feature/img': rutil.bytes_feature(img.tostring()),
        'feature/label': rutil.int64_feature(int(label))
    }))
    return example

def conver2Record2(dataset_path, output, cnt):
    dataset_subjects = ['lip_a', 'lip_n', 'face', 'background']
    img_paths = []
    labels = []
    for subject in dataset_subjects:
        subject_path = os.path.join(dataset_path, subject)

        if subject == 'lip_a' or subject == 'lip_n':
            cut_range = cnt
        else:
            cut_range = cnt * 2
        img_names = os.listdir(subject_path)[:cut_range]
        tem_paths = []
        for img_name in img_names:
            tem_paths.append(os.path.join(subject_path, img_name))
        if subject == 'lip_a' or subject == 'lip_n':
            img_labels = [0] * len(tem_paths)
        elif subject == 'face':
            img_labels = [1] * len(tem_paths)
        else:
            img_labels = [2] * len(tem_paths)
        img_paths = img_paths + tem_paths
        labels = labels + img_labels

    index_shuf = list(range(len(img_paths)))
    shuffle(index_shuf)
    shuffle_imgPaths = list()
    shuffle_labels = list()
    for i in index_shuf:
        shuffle_imgPaths.append(img_paths[i])
        shuffle_labels.append(labels[i])
    output_dir = os.path.dirname(output)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    with tf.python_io.TFRecordWriter(output) as tfWriter:
        for index in range(len(shuffle_imgPaths)):
            file = shuffle_imgPaths[index]
            label = shuffle_labels[index]
            img = cv2.imread(file)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            print('index:{}, file:{}, label:{}'.format(index, file, label))
            example = conver2Example(img, label)
            tfWriter.write(example.SerializeToString())


conver2Record('D:\\Development\\discri_lip_data\\test', 'train_pal_lfg_rgb.tfrecord', 200)


conver2Record2('D:\\Development\\discri_lip_test_data\\test', 'test_pal_lfg_rgb.tfrecord', 500)

