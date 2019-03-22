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
# Data set base class.
#
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod
import os

import tensorflow as tf

from utils import logger

log = logger.get()


class Dataset(object):
    """A simple class for handling data sets."""

    def __init__(self, name, subset, data_dir):
        """
        Initializes dataset using a subset and the path to the data.

        :param name:        [string]  Name of the data set.
        :param subset:      [string]  Name of the subset.
        :param data_dir:    [string]  Path to the data directory.
        """
        assert subset in self.available_subsets(), self.available_subsets()
        self._name = name
        self._subset = subset
        self._data_dir = data_dir

    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    def data_files(self):
        """
        Returns a python list of all (sharded) data subset files.

        :return             [list]  python list of all (sharded) data set files.
        """
        tf_record_pattern = os.path.join(self.data_dir, '%s-*' % self.subset)
        data_files_list = tf.gfile.Glob(tf_record_pattern)
        if not data_files_list:
            log.error('No files found for dataset {}/{} at {}'.format(self.name, self.subset,
                                                                      self.data_dir))
            self.download_message()
            exit(-1)
        return data_files_list

    def reader(self):
        """
        Returns reader for a single entry from the data set.

        :return            [object]  Reader object that reads the data set.
        """
        return tf.TFRecordReader()

    @property
    def name(self):
        """Returns the name of the dataset."""
        return self._name

    @property
    def subset(self):
        """Returns the name of the subset."""
        return self._subset

    @property
    def data_dir(self):
        """Returns the directory path of the dataset."""
        return self._data_dir
