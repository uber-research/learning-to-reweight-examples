# Modifications Copyright (c) 2019 Uber Technologies, Inc.
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Converts CIFAR datasets to TFRecord format and manually add flip-label noise.
# WARNING: the generated data is corrupted. Use it only for noisy label problem research.
#
# Output TFRecord in the following location:
# [output folder]/train-00000-of-00005
# [output folder]/train-00001-of-00005
# ...
# [output folder]/train-00004-of-00005
# [output folder]/validation-00000-of-00001
#
# Usage:
# ./generate_noisy_cifar_data.py --dataset       [DATASET NAME]            \
#                                --data_folder   [DATA FOLDER PATH]        \
#                                --seed          [RANDOM SEED]             \
#                                --noise_ratio   [NOISE RATIO]             \
#                                --num_clean     [NUM_CLEAN]               \
#                                --num_val       [NUM_VAL]
#
# Flags:
#   --data_folder:       Folder of where the dataset is stored.
#   --dataset:           Dataset name, `cifar-10` or `cifar-100`.
#   --noise_ratio:       Proportion of the noisy label data, default 0.01.
#   --seed:              Random seed to be used for clean/noisy split.
#   --num_clean:         Number of the clean training data, default 100.
#   --num_val:           Number of validation images, default 5000.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle as pkl
import six
import sys
import tensorflow as tf


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image, label, clean, idx):
    """Convert an image and label pair into a tf.train.Example.

    :param image:           [ndarray]      Image data, [H, W, C].
    :param label:           [int]          Label class ID.
    :param clean:           [int]          Whether the label is clean or noisy.
    :param idx:             [int]          Index of the example.

    :return                 [Example]      TF record item.
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'clean': _int64_feature(clean),
        'index': _int64_feature(idx)
    }))
    return example


def _unpickle(filename):
    """Reads in a pickle file."""
    with open(filename, 'rb') as fo:
        if sys.version_info[0] == 2:
            data_dict = pkl.load(fo)
        else:
            data_dict = pkl.load(fo, encoding='bytes')
    return data_dict


def _split(num, seed, partitions):
    """Split the training set into several partitions.

    :param num:             [int]          Total size of the training set.
    :param seed:            [int]          Random seed for creating the split.
    :param partitions:      [list]         List of integer indicating the partition size.

    :return:                [list]         List of numpy arrays.
    """
    all_idx = np.arange(num)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(all_idx)
    siz = 0
    results = []
    for pp in partitions:
        results.append(all_idx[siz:siz + pp])
        siz += pp
    return results


def serialize_to_tf_record(basename, num_shard, images, labels, mask=None):
    """Serializes to TF Record.

    :param basename:        [string]       Basename of the output TF records.
    :param num_shard:       [int]          Number of shards of TF records.
    :param images:          [ndarray]      Input image data, [N, H, W, C].
    :param labels:          [ndarray]      Label class ID data, [N].
    :param mask:            [ndarray]      Binary flag indicating whether the label is clean.
    """
    output_filename = basename + '-{:05d}-of-{:05d}'
    num_example = images.shape[0]
    num_example_per_shard = int(np.ceil(num_example / float(num_shard)))
    for ii in six.moves.xrange(num_shard):
        _filename = output_filename.format(ii, num_shard)
        with tf.python_io.TFRecordWriter(_filename) as writer:
            start = num_example_per_shard * ii
            end = min(num_example_per_shard * (ii + 1), num_example)
            for jj in six.moves.xrange(start, end):
                if mask is None:
                    _mask = 1
                else:
                    _mask = mask[jj]
                _example = _convert_to_example(images[jj].tobytes(), labels[jj], _mask, jj)
                writer.write(_example.SerializeToString())
            writer.flush()


def read_cifar_10(data_folder):
    """Reads and parses examples from CIFAR10 data files.

    :param data_folder:     [string]       Folder where the raw data are stored.

    :return                 [tuple]        A tuple of train images and labels, and test images and
                                           labels
    """
    train_file_list = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'
    ]
    test_file_list = ['test_batch']
    data_dict = {}
    for file_list, name in zip([train_file_list, test_file_list], ['train', 'validation']):
        img_list = []
        label_list = []
        for ii in six.moves.xrange(len(file_list)):
            data_dict = _unpickle(os.path.join(data_folder, 'cifar-10-batches-py', file_list[ii]))
            _img = data_dict[b'data']
            _label = data_dict[b'labels']
            _img = _img.reshape([-1, 3, 32, 32])
            _img = _img.transpose([0, 2, 3, 1])
            img_list.append(_img)
            label_list.append(_label)
        img = np.concatenate(img_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        if name == 'train':
            train_img = img
            train_label = label
        else:
            test_img = img
            test_label = label
    return train_img, train_label, test_img, test_label


def read_cifar_100(data_folder):
    """Reads and parses examples from CIFAR100 data files.

    :param data_folder:     [string]       Folder where the raw data are stored.

    :return                 [tuple]        A tuple of train images and labels, and test images and
                                           labels
    """
    train_file_list = ['train']
    test_file_list = ['test']

    data_dict = _unpickle(os.path.join(data_folder, 'cifar-100-python', train_file_list[0]))
    train_img = data_dict['data']
    train_label = np.array(data_dict['fine_labels'])

    data_dict = _unpickle(os.path.join(data_folder, 'cifar-100-python', test_file_list[0]))
    test_img = data_dict['data']
    test_label = np.array(data_dict['fine_labels'])

    train_img = train_img.reshape([-1, 3, 32, 32])
    train_img = train_img.transpose([0, 2, 3, 1])
    test_img = test_img.reshape([-1, 3, 32, 32])
    test_img = test_img.transpose([0, 2, 3, 1])
    return train_img, train_label, test_img, test_label


def trainval_split(img, label, num_val, seed):
    """Splits training set and validation set.

    :param img              [ndarray]      All images.
    :param label            [ndarray]      All labels.
    :param num_val          [int]          Number of validation images.
    :param seed             [int]          Random seed for generating the split.

    :return
    """
    assert img.shape[0] == label.shape[0], 'Images and labels dimension must match.'

    num = img.shape[0]
    trainval_partition = [num - num_val, num_val]
    idx = _split(num, seed, trainval_partition)

    return img[idx[0]], label[idx[0]], img[idx[1]], label[idx[1]]


def _flip_data(img, label, noise_ratio, num_classes, seed):
    """Flips label.

    :param img
    :param label
    :param noise_ratio
    :param num_clean
    :param num_classes
    :param seed

    :return                 [tuple]        A tuple of train images
    """
    # Flip labels in noise.
    num = len(label)
    assert len(img) == len(label)
    num_noise = int(num * noise_ratio)
    rnd = np.random.RandomState(seed + 1)
    noise_label_ = label[:num_noise]

    # Randomly re-assign labels.
    new_label = np.floor(rnd.uniform(0, num_classes - 1, [num_noise])).astype(np.int64)
    # new_noise_label_ = new_label + (new_label >= noise_label_).astype(np.int64)
    new_noise_label_ = new_label    # Not garanteeing flip.
    # assert not (new_noise_label_ == noise_label_).all(), 'New label is not corrupted.'

    label = np.concatenate([new_noise_label_, label[num_noise:]])
    noise_mask = np.concatenate([np.zeros([num_noise]), np.ones([num - num_noise])])
    noise_mask = noise_mask.astype(np.int64)

    # Re-shuffle again to mix in noisy examples.
    idx = np.arange(num)
    rnd.shuffle(idx)
    noise_mask = noise_mask[idx]
    noise_img = img[idx]
    noise_label = label[idx]
    return noise_img, noise_label, noise_mask


def _flip_data_background(img, label, noise_ratio, num_classes, seed):
    """Flips all the labels to another class.

    :param img
    :param label
    :param noise_ratio
    :param num_classes
    :param seed
    """
    # Flip labels in noise.
    num = len(label)
    assert len(img) == len(label)
    num_noise = int(num * noise_ratio)
    rnd = np.random.RandomState(seed + 1)
    noise_label_ = label[:num_noise]

    # Randomly re-assign labels.
    new_label = np.floor(rnd.uniform(0, num_classes - 1)).astype(np.int64)
    print('Random new label:', new_label)
    new_noise_label_ = np.zeros([num_noise], dtype=np.int64) + new_label
    noise_mask0 = (new_label == label[:num_noise]).astype(np.int64)

    label = np.concatenate([new_noise_label_, label[num_noise:]])
    noise_mask = np.concatenate([noise_mask0, np.ones([num - num_noise])])
    noise_mask = noise_mask.astype(np.int64)

    # Re-shuffle again to mix in noisy examples.
    idx = np.arange(num)
    rnd.shuffle(idx)
    noise_mask = noise_mask[idx]
    noise_img = img[idx]
    noise_label = label[idx]
    return noise_img, noise_label, noise_mask


def generate_data(img, label, noise_ratio, num_clean, num_classes, seed, background=False):
    """Generates noisy data.

    :param img              [ndarray]      Training images.
    :param label            [ndarray]      Training labels.
    :param noise_ratio      [float]        Noisy data ratio.
    :param num_clean        [int]          Number of clean images.
    :param num_classes      [int]          Number of classes.
    :param seed             [int]          Random seed for generating the split.

    :return                 [tuple]        A tuple of train images and labels and noise masks, and
                                           clean images and clean labels.
    """
    num = img.shape[0]
    noise_img, noise_label, clean_img, clean_label = trainval_split(img, label, num_clean, seed)
    if background:
        noise_img, noise_label, noise_mask = _flip_data_background(noise_img, noise_label,
                                                                   noise_ratio, num_classes, seed)
    else:
        noise_img, noise_label, noise_mask = _flip_data(noise_img, noise_label, noise_ratio,
                                                        num_classes, seed)
    return noise_img, noise_label, noise_mask, clean_img, clean_label


def generate_noisy_cifar(dataset,
                         data_folder,
                         num_val,
                         noise_ratio,
                         num_clean,
                         output_folder,
                         seed,
                         background=False):
    """Generates noisy cifar data and write TF records to disk.
    The output tf record has the following naming:
    1. Clean training set:    train_clean-?????-of-?????
    2. Noisy training set:    train_noisy-?????-of-?????
    3. Validation set:        validation-?????-of-?????
    4. Test set:              test-?????-of-?????

    :param dataset          [string]       Dataset name, `cifar-10` or `cifar-100`.
    :param data_folder      [string]       Data folder.
    :param num_val          [int]          Number of images in validation.
    :param noise_ratio      [float]        Ratio of noisy data in train_noisy.
    :param num_clean        [int]          Number of images in train_clean.
    :param output_folder    [string]       Path to the output folder.
    :param seed             [int]          Random seed for creating the split.
    """
    # Read in dataset.
    if dataset == 'cifar-10':
        train_img, train_label, test_img, test_label = read_cifar_10(data_folder)
        num_classes = 10
    elif dataset == 'cifar-100':
        train_img, train_label, test_img, test_label = read_cifar_100(data_folder)
        num_classes = 100

    # Split training set and validation set.
    train_img, train_label, val_img, val_label = trainval_split(train_img, train_label, num_val,
                                                                seed)
    # Split data and generate synthetic noise.
    noise_img, noise_label, noise_mask, clean_img, clean_label = generate_data(
        train_img, train_label, noise_ratio, num_clean, num_classes, seed, background=background)

    # Create output folder.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Noisy training set.
    serialize_to_tf_record(
        os.path.join(output_folder, 'train_noisy'), 4, noise_img, noise_label, noise_mask)

    # Clean training set.
    serialize_to_tf_record(os.path.join(output_folder, 'train_clean'), 1, clean_img, clean_label)

    # Validation set.
    serialize_to_tf_record(os.path.join(output_folder, 'validation'), 1, val_img, val_label)

    # Contaminate validation set.
    if background:
        noise_val_img, noise_val_label, noise_val_mask = _flip_data_background(
            val_img, val_label, noise_ratio, num_classes, seed)
    else:
        noise_val_img, noise_val_label, noise_val_mask = _flip_data(val_img, val_label, noise_ratio,
                                                                    num_classes, seed)

    # Validation set.
    serialize_to_tf_record(
        os.path.join(output_folder, 'validation_noisy'), 1, noise_val_img, noise_val_label,
        noise_val_mask)

    # Test set.
    serialize_to_tf_record(os.path.join(output_folder, 'test'), 1, test_img, test_label)


def main():
    generate_noisy_cifar(FLAGS.dataset, FLAGS.data_folder, FLAGS.num_val, FLAGS.noise_ratio,
                         FLAGS.num_clean, FLAGS.output_folder, FLAGS.seed)


if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_float('noise_ratio', 0.4, 'Ratio of the noisy data, default 0.4')
    flags.DEFINE_integer('num_clean', 100, 'Number of clean training data, default 100')
    flags.DEFINE_integer('num_val', 5000, 'Number of validation data, default 5000')
    flags.DEFINE_integer('seed', 0, 'Random seed, default 0')
    flags.DEFINE_string('data_folder', './data/cifar-10',
                        'Data set folder, default `./data/cifar-10`')
    flags.DEFINE_string('dataset', 'cifar-10', 'Data set name, default `cifar-10`')
    flags.DEFINE_string('output_folder', './data/cifar-10-noisy',
                        'TFRecord output folder, default `./data/cifar-10-noisy')
    FLAGS = tf.flags.FLAGS
    main()
