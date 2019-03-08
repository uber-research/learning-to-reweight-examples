# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# A collection of optimizers.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def get_optimizer(optimizer_type):
    """Gets an optimizer."""
    if optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer
    else:
        raise ValueError('Unknown optimizer type: {}'.format(config.type))
