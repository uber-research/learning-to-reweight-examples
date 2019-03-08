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
# Residual networks model with externally assigned weights (parameters).
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from utils import logger
from models.cnn.cnn_model import CNNModel
from models.model_factory import RegisterModel
from models.cnn.resnet_module import ResnetModule
from models.base.nnlib import weight_variable_cpu, batch_norm

log = logger.get()


class AssignedWeightsResnetModule(ResnetModule):
    def __init__(self, config, weights_dict, is_training=True):
        """Initialize the module with a weight dictionary.

        :param config:          [object]  A configuration object.
        :param weights_dict:    [dict]    A dictionary that stores all the parameters.
        :param is_training:     [bool]    Whether in training mode.
        """
        self._weights_dict = weights_dict
        if weights_dict is None:
            self._create_new_var = True
        else:
            self._create_new_var = False
        super(AssignedWeightsResnetModule, self).__init__(config, is_training=is_training)

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """A wrapper to declare variables on CPU.
        If weights_dict is not None, it retrieves the historical weights.
        """
        # Here grab a shared variable first.
        var = super(AssignedWeightsResnetModule, self)._weight_variable(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=0)
        if self.create_new_var:
            if self.weights_dict is None:
                self.weights_dict = {}
            self.weights_dict[var.name] = var
            return var
        else:
            return self.weights_dict[var.name]

    def _batch_norm(self, name, x):
        """Batch normalization."""
        if self.data_format == 'NCHW':
            axis = 1
            axes = [0, 2, 3]
        else:
            axis = -1
            axes = [0, 1, 2]
        with tf.variable_scope('BatchNorm'):
            beta = self._weight_variable([int(x.get_shape()[axis])], name='beta')
        mean, var = tf.nn.moments(x, axes=axes)
        if self.data_format == 'NCHW':
            beta = tf.reshape(beta, [1, -1, 1, 1])
            mean = tf.reshape(mean, [1, -1, 1, 1])
            var = tf.reshape(var, [1, -1, 1, 1])
        return tf.nn.batch_normalization(x, mean, var, beta, None, 0.001)

    @property
    def create_new_var(self):
        return self._create_new_var

    @property
    def weights_dict(self):
        return self._weights_dict

    @weights_dict.setter
    def weights_dict(self, weights_dict):
        self._weights_dict = weights_dict


@RegisterModel('assign-wts-resnet')
class AssignedWeightsResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self,
                 config,
                 weights_dict,
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
        if ex_wts is None:
            w = tf.placeholder(self.dtype, [batch_size], 'w')
        else:
            w = ex_wts
        self._ex_wts = w
        super(AssignedWeightsResnetModel, self).__init__(
            config,
            AssignedWeightsResnetModule(
                config.resnet_module_config, weights_dict, is_training=is_training),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)

    @classmethod
    def create_from_file(cls,
                         config_filename,
                         is_training=True,
                         inp=None,
                         label=None,
                         batch_size=None):
        config = ResnetModelConfig()
        Merge(open(config_filename).read(), config)
        return cls(config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [Tensor]    Output of the network.

        :return                 [Scalar]    Loss value.
        """
        with tf.variable_scope('costs'):
            label = tf.one_hot(self.label, self._cnn_module.config.num_classes)
            # xent = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.label)
            # xent = tf.losses.softmax_cross_entropy(label, output)
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
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
