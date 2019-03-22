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
# Data input pipeline base class for TF Records.
#
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod

import tensorflow as tf


class InputPipeline(object):
    """TFRecord input pipeline."""

    def __init__(self, dataset, is_training, batch_size):
        self._dataset = dataset
        self._is_training = is_training
        self._batch_size = batch_size

    def inputs(self,
               num_epochs=None,
               num_preprocess_threads=4,
               num_readers=None,
               queue_memory_factor=16,
               filename_queue_size=None,
               seed=0):
        """
        Generates batches inputs.

        :param num_epoch:              [int]  Number of epochs.
        :param num_preprocess_threads: [int]  Number of workers, default 4
        :param num_readers:            [int]  Number of parallel data readers, default 4 for
                                              training and 1 for evaluation.
        :param queue_memory_factor:    [int]  Memory queue size ratio to number of examples per
                                              shard.
        :param filename_queue_size:    [int]  Size of the filename queue.
        :param seed:                   [int]  Random seed for shuffling data.

        :return:                       [dict] A batched data input dictionary.
        """
        if num_readers is None:
            num_readers = 4 if self.is_training else 1
        if filename_queue_size is None:
            filename_queue_size = 16 if self.is_training else 1
        # Force all input processing onto CPU in order to reserve the GPU for
        # the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            example_batch = self._inputs(
                num_epochs=num_epochs,
                num_preprocess_threads=num_preprocess_threads,
                num_readers=num_readers,
                queue_memory_factor=queue_memory_factor,
                filename_queue_size=filename_queue_size,
                seed=seed)
        return example_batch

    @abstractmethod
    def parse_example_proto(self, example_serialized):
        """Parses an Example proto."""
        pass

    @abstractmethod
    def preprocess_example(self, example, is_training, thread_id=0):
        """Input preprocessing."""
        pass

    def _inputs(self, num_epochs, num_preprocess_threads, num_readers, queue_memory_factor,
                filename_queue_size, seed):
        """
        Contructs batches of training or evaluation examples from the image dataset.

        :param num_epoch:              [int]  Number of epochs.
        :param num_preprocess_threads: [int]  Number of preprocessing workers.
        :param num_readers:            [int]  Number of parallel readers.
        :param queue_memory_factor:    [int]  Memory queue size ratio to number of examples per
                                              shard.
        :param filename_queue_size:    [int]  Size of the filename queue.
        :param seed:                   [int]  Random seed for shuffling data.

        :return:                       [dict] A batched data input dictionary.
        """
        dataset = self.dataset
        batch_size = self.batch_size
        is_training = self.is_training
        with tf.name_scope('batch_processing'):
            data_files = self.dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            # Create filename_queue
            filename_queue = tf.train.string_input_producer(
                data_files,
                shuffle=is_training,
                capacity=filename_queue_size,
                num_epochs=num_epochs,
                seed=seed)

            if num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                                 'of 4 (%d % 4 != 0).', num_preprocess_threads)

            if num_readers < 1:
                raise ValueError('Please make num_readers at least 1')

            # Approximate number of examples per shard.
            examples_per_shard = dataset.num_examples_per_epoch() // len(data_files)
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = examples_per_shard * queue_memory_factor

            if is_training:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string],
                    seed=seed)
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            examples = []

            if num_preprocess_threads > 0:
                for thread_id in range(num_preprocess_threads):
                    # Parse a serialized Example proto to extract the image and metadata.
                    example = self.parse_example_proto(example_serialized)
                    example = self.preprocess_example(example, is_training, thread_id)
                    examples.append(example)
                example_batch = tf.train.batch_join(
                    examples,
                    batch_size=batch_size,
                    capacity=2 * num_preprocess_threads * batch_size,
                    allow_smaller_final_batch=True)
            else:
                example = self.parse_example_proto(example_serialized)
                example = self.preprocess_example(example, is_training)
                examples.append(example)

                example_batch = tf.train.batch_join(
                    examples,
                    batch_size=batch_size,
                    capacity=2 * batch_size,
                    allow_smaller_final_batch=True)

            return example_batch

    @property
    def dataset(self):
        return self._dataset

    @property
    def is_training(self):
        return self._is_training

    @property
    def batch_size(self):
        return self._batch_size
