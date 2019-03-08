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
# Implements paper J. Goldberger, E. Ben-Reuven. "Training deep neural networks using a noise
# adaptation layer". ICLR. 2017.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from models.cnn.resnet_module import ResnetModule
from models.cnn.resnet_model import ResnetModel
from models.cnn.resnet_model import CNNModel
from models.model_factory import RegisterModel
from models.base.nnlib import batch_norm


class NoiseAdaptResnetModule(ResnetModule):
    def __init__(self, config, confusion_matrix, is_training=True):
        """A resnet module with a noise adapt layer after softmax.
        
        :param config:             [object]      A configuration object.
        :param confusion_matrix:   [np.ndarray]  Confusion matrix.
        :param is_training:        [bool]        Whether in training mode.
        """
        self._confusion_matrix = confusion_matrix
        super(NoiseAdaptResnetModule, self).__init__(config, is_training=is_training)

    def __call__(self, inp):
        h = super(NoiseAdaptResnetModule, self).__call__(inp)
        if self._is_training:
            y = tf.nn.softmax(h)
            # Initialize with identity.
            if self._confusion_matrix is not None:
                init = self._confusion_matrix.astype(np.float32) / self._confusion_matrix.sum(
                    axis=1, keepdims=True)
                print('confusion matrix initialization')
                print(init)
                init += 1e-3
                init = tf.log(tf.constant(init, dtype=tf.float32))
            else:
                init = np.eye(self.config.num_classes, dtype=np.float32)
                init += 1e-3
                init = tf.log(tf.constant(init, dtype=tf.float32))
            w_noise = tf.get_variable('w_noise', dtype=tf.float32, initializer=init)
            w_noise_norm = tf.nn.softmax(w_noise)
            z = tf.matmul(y, w_noise_norm)
            return tf.log(z)


@RegisterModel('noise-adapt-resnet')
class NoiseAdaptResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self,
                 config,
                 confusion_matrix,
                 is_training=True,
                 inp=None,
                 label=None,
                 ex_wts=None,
                 batch_size=None):
        """
        Resnet constructor.

        :param config:      [object]    Configuration object.
        :param weights_dict [dict]      Dictionary for assigned weights.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        # Example weights.
        super(NoiseAdaptResnetModel, self).__init__(
            config,
            NoiseAdaptResnetModule(
                config.resnet_module_config, confusion_matrix, is_training=is_training),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)
