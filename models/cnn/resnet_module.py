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
# Residual networks module.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from models.base.nnlib import concat, weight_variable_cpu, batch_norm
from utils.logger import get as get_logger

log = get_logger()


class ResnetModule(object):
    """Resnet module."""

    def __init__(self, config, is_training=True):
        """
        Resnet module constructor.

        :param config:      [object]    Configuration object, see configs/resnet_module_config.proto.
        """
        self._config = config
        self._dtype = tf.float32
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
        if len(config.dilations) == 0:
            dilations = [1] * len(config.strides)
        else:
            dilations = config.dilations
        assert len(config.strides) == len(dilations), 'Need to pass in lists of same size.'
        filters = [ff for ff in config.num_filters]    # Copy filter config.
        init_filter_size = config.init_filter_size
        if self.data_format == 'NCHW':
            inp = tf.transpose(inp, [0, 3, 1, 2])

        with tf.variable_scope('init'):
            h = self._conv('init_conv', inp, init_filter_size, self.config.num_channels, filters[0],
                           self._stride_arr(config.init_stride), 1)
            h = self._batch_norm('init_bn', h)
            h = self._relu('init_relu', h)

            # Max-pooling is used in ImageNet experiments to further reduce
            # dimensionality.
            if config.init_max_pool:
                h = tf.nn.max_pool(
                    h,
                    self._stride_arr(3),
                    self._stride_arr(2),
                    'SAME',
                    data_format=self.data_format)

        if config.use_bottleneck:
            res_func = self._bottleneck_residual
            # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
            for ii in range(1, len(filters)):
                filters[ii] *= 4
        else:
            res_func = self._residual

        # New version, single for-loop. Easier for checkpoint.
        nlayers = sum(config.num_residual_units)
        ss = 0
        ii = 0
        for ll in range(nlayers):
            # Residual unit configuration.
            if ii == 0:
                if ss == 0:
                    no_activation = True
                else:
                    no_activation = False
                in_filter = filters[ss]
                stride = self._stride_arr(strides[ss])
            else:
                in_filter = filters[ss + 1]
                stride = self._stride_arr(1)

            # Compute out filters.
            out_filter = filters[ss + 1]

            # Compute dilation rates.
            if dilations[ss] > 1:
                if config.use_bottleneck:
                    dilation = [dilations[ss] // strides[ss], dilations[ss], dilations[ss]]
                else:
                    dilation = [dilations[ss] // strides[ss], dilations[ss]]
            else:
                if config.use_bottleneck:
                    dilation = [1, 1, 1]
                else:
                    dilation = [1, 1]

            # Build residual unit.
            with tf.variable_scope('unit_{}_{}'.format(ss + 1, ii)):
                h = res_func(
                    h,
                    in_filter,
                    out_filter,
                    stride,
                    dilation,
                    dropout=dropout,
                    no_activation=no_activation)

            if (ii + 1) % config.num_residual_units[ss] == 0:
                ss += 1
                ii = 0
            else:
                ii += 1

        # Make a single tensor.
        if type(h) == tuple:
            h = concat(h, axis=3)

        # Classification layer.
        if config.build_classifier:
            with tf.variable_scope('unit_last'):
                h = self._batch_norm('final_bn', h)
                h = self._relu('final_relu', h)

            h = self._global_avg_pool(h)
            with tf.variable_scope('logit'):
                h = self._fully_connected(h, config.num_classes)

        return h

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

    def _stride_arr(self, stride):
        """
        Map a stride scalar to the stride array for tf.nn.conv2d.

        :param stride: [int] Size of the stride.

        :return:       [list] [1, stride, stride, 1]
        """
        if self.data_format == 'NCHW':
            return [1, 1, stride, stride]
        else:
            return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """
        Applies batch normalization.

        :param name:    [string]    Name of the variable scope.
        :param x:       [Tensor]    Tensor to apply BN on.
        :param add_ops: [bool]      Whether to add BN updates to the ops list, default True.

        :return:        [Tensor]    Normalized activation.
        """
        bn = tf.contrib.layers.batch_norm(
            x, fused=True, data_format=self.data_format, is_training=self.is_training)
        return bn

    def _possible_downsample(self, x, in_filter, out_filter, stride):
        """
        Downsamples the feature map using average pooling, if the filter size
        does not match.

        :param x:             [Tensor]     Input to the downsample.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.

        :return:              [Tensor]     Possibly downsampled activation.
        """
        if stride[2] > 1:
            with tf.variable_scope('downsample'):
                x = tf.nn.avg_pool(x, stride, stride, 'SAME', data_format=self.data_format)

        if in_filter < out_filter:
            with tf.variable_scope('pad'):
                pad_ = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                if self.data_format == 'NCHW':
                    x = tf.pad(x, [[0, 0], pad_, [0, 0], [0, 0]])
                else:
                    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_])
        return x

    def _residual_inner(self,
                        x,
                        in_filter,
                        out_filter,
                        stride,
                        dilation_rate,
                        dropout=0.0,
                        no_activation=False):
        """
        Inner transformation applied on residual units.

        :param x:              [Tensor]     Input to the residual function.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate:  [list]       List of 2 integers, dilation rate for each conv.
        :param dropout:        [float]      Whether to dropout in the middle.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [Tensor]     Output of the residual function.
        """
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
                if dropout > 0.0 and self.is_training:
                    log.info('Using dropout with {:d}%'.format(int(dropout * 100)))
                    x = tf.nn.dropout(x, keep_prob=(1.0 - dropout))
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter, out_filter,
                           self._stride_arr(1), dilation_rate[1])
        return x

    def _residual(self,
                  x,
                  in_filter,
                  out_filter,
                  stride,
                  dilation_rate,
                  dropout=0.0,
                  no_activation=False):
        """
        A residual unit with 2 sub layers.

        :param x:              [tf.Tensor]  Input to the residual unit.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate   [list]       List of 2 integers, dilation rate for each conv.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [tf.Tensor]  Output of the residual unit.
        """
        orig_x = x
        x = self._residual_inner(
            x,
            in_filter,
            out_filter,
            stride,
            dilation_rate,
            dropout=dropout,
            no_activation=no_activation)
        x += self._possible_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _bottleneck_residual_inner(self,
                                   x,
                                   in_filter,
                                   out_filter,
                                   stride,
                                   dilation_rate,
                                   no_activation=False):
        """
        Inner transformation applied on residual units (bottleneck).

        :param x:              [Tensor]     Input to the residual function.
        :param in_filter:      [int]        Input number of channels.
        :param out_filter:     [int]        Output number of channels.
        :param stride:         [list]       4-D strides array.
        :param dilation_rate   [list]       List of 3 integers, dilation rate for each conv.
        :param no_activation:  [bool]       Whether to have BN+ReLU in the first.

        :return:               [Tensor]     Output of the residual function.
        """
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
            x = self._conv('conv1', x, 1, in_filter, out_filter // 4, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter // 4, out_filter // 4,
                           self._stride_arr(1), dilation_rate[1])
        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu('relu3', x)
            x = self._conv('conv3', x, 1, out_filter // 4, out_filter,
                           self._stride_arr(1), dilation_rate[2])
        return x

    def _possible_bottleneck_downsample(self, x, in_filter, out_filter, stride):
        """Downsample projection layer, if the filter size does not match.

        :param x:             [Tensor]     Input to the downsample.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.
        :param dilation_rate: [int]        Dilation rate.

        :return:              [Tensor]     Possibly downsampled activation.
        """
        if stride[1] > 1 or in_filter != out_filter:
            x = self._conv('project', x, 1, in_filter, out_filter, stride, 1)
        return x

    def _bottleneck_residual(self,
                             x,
                             in_filter,
                             out_filter,
                             stride,
                             dilation_rate,
                             no_activation=False):
        """
        A bottleneck resisual unit with 3 sub layers.

        :param x:              [Tensor]    Input to the residual unit.
        :param in_filter:      [int]       Input number of channels.
        :param out_filter:     [int]       Output number of channels.
        :param stride:         [list]      4-D strides array.
        :param dilation_rate   [list]      List of 3 integers, dilation rate for each conv.
        :param no_activation:  [bool]      Whether to have BN+ReLU in the first.

        :return:               [Tensor]    Output of the residual unit.
        """
        orig_x = x
        x = self._bottleneck_residual_inner(
            x, in_filter, out_filter, stride, dilation_rate, no_activation=no_activation)
        x += self._possible_bottleneck_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, dilation_rate):
        """Convolution.

        :param name           [string]     Name of the op.
        :param x:             [Tensor]     Input to the downsample.
        :param filter_size    [list]       4-D kernel shape.
        :param in_filter:     [int]        Input number of channels.
        :param out_filter:    [int]        Output number of channels.
        :param stride:        [list]       4-D strides array.
        :param dilation_rate: [int]        Convolution dilation rate.

        :return:              [Tensor]     Convolution output.
        """
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            init_method = 'truncated_normal'
            init_param = {'mean': 0, 'stddev': np.sqrt(2.0 / n)}
            kernel = self._weight_variable(
                [filter_size, filter_size, in_filters, out_filters],
                init_method=init_method,
                init_param=init_param,
                weight_decay=self.config.weight_decay,
                dtype=self.dtype,
                name='w')
            if dilation_rate == 1:
                return tf.nn.conv2d(
                    x, kernel, strides, padding='SAME', data_format=self.data_format)
            elif dilation_rate > 1:
                assert self.data_format == 'NHWC', 'Dilated convolution needs to be in NHWC format.'
                assert all([strides[ss] == 1 for ss in range(len(strides))]), 'Strides need to be 1'
                return tf.nn.atrous_conv2d(x, kernel, dilation_rate, padding='SAME')

    def _relu(self, name, x):
        """
        Applies ReLU function.

        :param name: [string]     Name of the op.
        :param x:    [Tensor]     Input to the function.

        :return:     [Tensor]     Output of the function.
        """
        return tf.nn.relu(x, name=name)

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

    def _global_avg_pool(self, x):
        """
        Applies global average pooling.

        :param x:  [Tensor]   Input to average pooling.

        :return:   [Tensor]   Pooled activation.
        """
        if self.data_format == 'NCHW':
            return tf.reduce_mean(x, [2, 3])
        else:
            return tf.reduce_mean(x, [1, 2])

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_format(self):
        return self._data_format

    @property
    def is_training(self):
        return self._is_training
