#
# Uber, Inc. (c) 2017
#
# CIFAR data sets.
#
from __future__ import absolute_import, division, print_function

from datasets.base.dataset import Dataset
from datasets.data_factory import RegisterDataset


class NoisyCifarDataset(Dataset):
    """CIFAR data set."""

    def __init__(self, name, subset, data_dir, num_clean, num_val):
        super(NoisyCifarDataset, self).__init__(name, subset, data_dir)
        self._num_clean = num_clean
        self._num_val = num_val

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train_noisy':
            return 50000 - self.num_val - self.num_clean
        elif self.subset == 'train_clean':
            return self.num_clean
        elif self.subset in ['validation', 'validation_noisy']:
            return self.num_val
        elif self.subset == 'test':
            return 10000

    def download_message(self):
        pass

    def available_subsets(self):
        return ['train_noisy', 'train_clean', 'validation', 'validation_noisy', 'test']

    @property
    def num_clean(self):
        return self._num_clean

    @property
    def num_val(self):
        return self._num_val


@RegisterDataset("cifar-10-noisy")
class NoisyCifar10Dataset(NoisyCifarDataset):
    """CIFAR-10 data set."""

    def __init__(self, subset, data_dir, num_clean, num_val):
        super(NoisyCifar10Dataset, self).__init__('Noisy CIFAR-10', subset, data_dir, num_clean,
                                                  num_val)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10


@RegisterDataset("cifar-100-noisy")
class NoisyCifar100Dataset(NoisyCifarDataset):
    """CIFAR-100 data set."""

    def __init__(self, subset, data_dir, num_clean, num_val):
        super(NoisyCifar100Dataset, self).__init__('Noisy CIFAR-100', subset, data_dir, num_clean,
                                                   num_val)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 100
