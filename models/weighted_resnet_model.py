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
# Residual networks model with non-uniform example weights.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from models.cnn.resnet_module import ResnetModule
from models.cnn.resnet_model import ResnetModel
from models.model_factory import RegisterModel
from models.base.nnlib import batch_norm


@RegisterModel('weighted-resnet')
class WeightedResnetModel(ResnetModel):
    """Resnet model."""

    def __init__(self, config, is_training=True, inp=None, label=None, ex_wts=None,
                 batch_size=None):
        """
        Resnet constructor.

        :param config:      [object]    Configuration object.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        # Example weights.
        if ex_wts is None:
            w = tf.placeholder(self.dtype, [batch_size], 'w')
        else:
            w = ex_wts
        self._ex_wts = w
        super(WeightedResnetModel, self).__init__(
            config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    def _get_feed_dict(self, inp=None, label=None, ex_wts=None):
        """
        Generates feed dict.

        :param inp:      [ndarray]    Optional inputs, for placeholder only.
        :param label:    [ndarray]    Optional labels, for placeholder only.

        :return:         [dict]       A dictionary from model Tensor to input numpy arrays.
        """
        if inp is None and label is None and ex_wts is None:
            return None
        feed_data = {}
        if inp is not None:
            feed_data[self.input] = inp
        if label is not None:
            feed_data[self.label] = label
        if ex_wts is not None:
            feed_data[self.ex_wts] = ex_wts
        return feed_data

    def train_step(self, sess, inp=None, label=None, ex_wts=None):
        """
        Runs one training step.

        :param sess:     [Session]    TensorFlow Session object.
        :param inp:      [ndarray]    Optional inputs, for placeholder only.
        :param label:    [ndarray]    Optional labels, for placeholder only.
        :param ex_wts:   [ndarray]    Optional weights for each example.

        :return:         [float]      Cross entropy value.
        """
        results = sess.run(
            [self.cross_ent, self.train_op] + self.bn_update_ops,
            feed_dict=self._get_feed_dict(inp=inp, label=label, ex_wts=ex_wts))
        return results[0]

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [Tensor]    Output of the network.

        :return                 [Scalar]    Loss value.
        """
        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.label)
            xent_avg = tf.reduce_mean(xent, name='xent')
            xent_wt = tf.reduce_sum(xent * self.ex_wts, name='xent_wt')
            cost = xent_wt
            cost += self._decay()
            self._cross_ent_avg = xent_avg
            self._cross_ent_wt = xent_wt
        return cost

    @property
    def ex_wts(self):
        return self._ex_wts

    @property
    def cross_ent(self):
        return self._cross_ent_avg
