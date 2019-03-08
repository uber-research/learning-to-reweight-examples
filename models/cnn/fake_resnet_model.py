# Copyright (c) 2019 Uber Technologies, Inc.
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
# A fake resnet model built for gradient testing. Using tf.float64.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from google.protobuf.text_format import Merge

from models.base.nnlib import weight_variable_cpu
from models.cnn.cnn_model import CNNModel
from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig
from models.model_factory import RegisterModel


class FakeResnetModule(object):
    """A fake resnet."""

    def __init__(self, config, is_training=True):
        """
        Resnet module constructor.

        :param config:      [object]    Configuration object, see configs/resnet_module_config.proto.
        """
        self._config = config
        self._data_format = config.data_format
        self._is_training = is_training

    def __call__(self, inp):
        """
        Builds Resnet graph.

        :param inp:         [Tensor]    Input tensor to the Resnet, [B, H, W, C].

        :return:            [Tensor]    Output tensor, [B, Ho, Wo, Co] if not build classifier,
                                        [B, K] if build classifier.
        """
        config = self.config
        strides = config.strides
        dropout = config.dropout
        filters = [ff for ff in config.num_filters]    # Copy filter config.
        init_filter_size = config.init_filter_size
        assert self.data_format == 'NHWC'
        inp = tf.reshape(inp, [inp.shape[0], -1])
        h = inp
        if config.build_classifier:
            with tf.variable_scope('logit'):
                h = self._fully_connected(h, config.num_classes)
        return h

    def _fully_connected(self, x, out_dim):
        """
        A FullyConnected layer for final output.

        :param x:         [Tensor]     Input to the fully connected layer.
        :param out_dim:   [int]        Number of output dimension.

        :return:          [Tensor]     Output of the fully connected layer.
        """
        x_shape = x.get_shape()
        d = x_shape[1]
        w = self._weight_variable(
            [d, out_dim],
            init_method='uniform_scaling',
            init_param={'factor': 1.0},
            weight_decay=self.config.weight_decay,
            dtype=self.dtype,
            name='w')
        b = self._weight_variable(
            [out_dim], init_method='constant', init_param={'val': 0.0}, name='b', dtype=self.dtype)
        return tf.nn.xw_plus_b(x, w, b)

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """
        A wrapper to declare variables on CPU.

        See nnlib.py:weight_variable_cpu for documentation.
        """
        return weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed)

    @property
    def config(self):
        return self._config

    @property
    def data_format(self):
        return self._data_format

    @property
    def dtype(self):
        return tf.float64


@RegisterModel('fake-resnet')
class FakeResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self, config, is_training=True, inp=None, label=None, batch_size=None):
        """
        Resnet constructor.

        :param config:      [object]    Configuration object.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        self._config = config
        self._is_training = is_training
        super(FakeResnetModel, self).__init__(
            config,
            self._get_resnet_module(),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)

    def _get_resnet_module(self):
        return FakeResnetModule(self.config.resnet_module_config, is_training=self.is_training)

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

    @property
    def dtype(self):
        return tf.float64


class FakeAssignedWeightsResnetModule(FakeResnetModule):
    def __init__(self, config, weights_dict, is_training=True):
        self._weights_dict = weights_dict
        if weights_dict is None:
            self._create_new_var = True
        else:
            self._create_new_var = False
        super(FakeAssignedWeightsResnetModule, self).__init__(config, is_training=is_training)

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
        var = super(FakeAssignedWeightsResnetModule, self)._weight_variable(
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


@RegisterModel('fake-assign-wts-resnet')
class FakeAssignedWeightsResnetModel(CNNModel):
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
        super(FakeAssignedWeightsResnetModel, self).__init__(
            config,
            FakeAssignedWeightsResnetModule(
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

    @property
    def dtype(self):
        return tf.float64
