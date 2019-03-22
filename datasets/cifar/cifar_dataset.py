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
# CIFAR data sets.
#
from __future__ import absolute_import, division, print_function

from datasets.base.dataset import Dataset
from datasets.data_factory import RegisterDataset


class CifarDataset(Dataset):
    """CIFAR data set."""

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'trainval', 'test']

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return 45000
        if self.subset == 'validation':
            return 5000
        if self.subset == 'trainval':
            return 50000
        if self.subset == 'test':
            return 10000

    def download_message(self):
        pass


@RegisterDataset("cifar-10")
class Cifar10Dataset(CifarDataset):
    """CIFAR-10 data set."""

    def __init__(self, subset, data_dir):
        super(Cifar10Dataset, self).__init__('CIFAR-10', subset, data_dir)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10


@RegisterDataset("cifar-100")
class Cifar100Dataset(CifarDataset):
    """CIFAR-100 data set."""

    def __init__(self, subset, data_dir):
        super(Cifar100Dataset, self).__init__('CIFAR-100', subset, data_dir)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 100
