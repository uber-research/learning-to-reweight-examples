#
# Uber, Inc. (c) 2017
#
# Implements paper "Training deep neural networks using a noise adaptation layer"
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from models.cnn.resnet_module import ResnetModule
from models.cnn.resnet_model import ResnetModel
from models.cnn.resnet_model import CNNModel
from models.model_factory import RegisterModel
from models.base.nnlib import batch_norm


@RegisterModel('reed-resnet')
class ReedResnetModel(ResnetModel):
    """Resnet model."""

    def __init__(self,
                 config,
                 beta,
                 is_hard,
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
        self._beta = beta
        self._is_hard = is_hard
        super(ReedResnetModel, self).__init__(
            config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [Tensor]    Output of the network.

        :return                 [Scalar]    Loss value.
        """
        with tf.variable_scope('costs'):
            label = tf.one_hot(self.label, self._cnn_module.config.num_classes)
            xent = self._beta * tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)

            if self._is_hard:
                prediction = tf.argmax(output, axis=1)
                label = tf.one_hot(prediction, self._cnn_module.config.num_classes)
                xent += (1.0 - self._beta) * tf.nn.softmax_cross_entropy_with_logits(
                    labels=label, logits=output)
            else:
                prediction = tf.nn.softmax(output)
                xent += (1.0 - self._beta) * tf.nn.softmax_cross_entropy_with_logits(
                    labels=prediction, logits=output)

            xent = tf.reduce_mean(xent, name='xent')
            cost = xent
            cost += self._decay()
            self._cross_ent = xent
        return cost
