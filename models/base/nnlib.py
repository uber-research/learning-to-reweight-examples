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
# Basic neural networks using TensorFlow.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from utils import logger

log = logger.get()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    weight_decay=None,
                    name=None,
                    trainable=True,
                    seed=0):
    """
    Declares a variable.

    :param shape:         [list]     Shape of the weights.
    :param init_method:   [string]   Initialization method, one of 'constant', 'truncated_normal',
                                     'uniform_scaling', 'xavier'.
    :param dtype          [dtype]    Data type, one of TensorFlow dtypes.
    :param init_param:    [dict]     Initialization parameters.
    :param weight_decay:  [float]    Weight decay.
    :param name:          [string]   Name of the variable.
    :param trainable:     [bool]     Whether the variable can be trained.
    :param seed:          [int]      Initialization seed.

    :return:              [tf.Variable] Declared variable.
    """
    if dtype != tf.float32:
        log.warning('Not using float32, currently using {}'.format(dtype))
    if init_method is None:
        initializer = tf.zeros_initializer(dtype=dtype)
    elif init_method == 'truncated_normal':
        if 'mean' not in init_param:
            mean = 0.0
        else:
            mean = init_param['mean']
        if 'stddev' not in init_param:
            stddev = 0.1
        else:
            stddev = init_param['stddev']
        log.info('Normal initialization std {:.3e}'.format(stddev))
        initializer = tf.truncated_normal_initializer(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)
    elif init_method == 'uniform_scaling':
        if 'factor' not in init_param:
            factor = 1.0
        else:
            factor = init_param['factor']
        log.info('Uniform initialization scale {:.3e}'.format(factor))
        initializer = tf.uniform_unit_scaling_initializer(factor=factor, seed=seed, dtype=dtype)
    elif init_method == 'constant':
        if 'val' not in init_param:
            value = 0.0
        else:
            value = init_param['val']
        initializer = tf.constant_initializer(value=value, dtype=dtype)
    elif init_method == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed, dtype=dtype)
    else:
        raise ValueError('Non supported initialization method!')
    try:
        shape_int = [int(ss) for ss in shape]
        log.info('Weight shape {}'.format(shape_int))
    except:
        pass
    if weight_decay is not None:
        if weight_decay > 0.0:

            def _reg(x):
                return tf.multiply(tf.nn.l2_loss(x), weight_decay)

            reg = _reg
            log.info('Weight decay {}'.format(weight_decay))
        else:
            reg = None
    else:
        reg = None
    var = tf.get_variable(
        name, shape, initializer=initializer, regularizer=reg, dtype=dtype, trainable=trainable)

    if weight_decay is None or weight_decay == 0.0:
        log.warning('No weight decay for {}'.format(var.name))

    log.info('Initialized weight {}'.format(var.name))
    return var


def weight_variable_cpu(shape,
                        init_method=None,
                        dtype=tf.float32,
                        init_param=None,
                        weight_decay=None,
                        name=None,
                        trainable=True,
                        seed=0):
    """
    Declares variables on CPU.
    See weight_variable for usage.
    """
    with tf.device('/cpu:0'):
        return weight_variable(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed)


def concat(x, axis):
    """
    Concatenates a list of tensors.

    :param x:     [list]   A list of TensorFlow tensor objects.
    :param axis:  [int]    Axis along which to be concatenated.

    :return:      [Tensor] Concatenated tensor.
    """
    if tf.__version__.startswith('0'):    # 0.12 compatibility.
        return tf.concat(axis, x)
    else:
        return tf.concat(x, axis=axis)


def split(x, num, axis):
    """
    Splits a tensor into a list of tensors.

    :param x:     [Tensor]  A TensorFlow tensor object to be split.
    :param num:   [int]     Number of splits.
    :param axis:  [int]     Axis along which to be split.

    :return:      [list]    A list of TensorFlow tensor objects.
    """
    if tf.__version__.startswith('0'):    # 0.12 compatibility.
        return tf.split(axis, num, x)
    else:
        return tf.split(x, num, axis)


def stack(x):
    """
    Stacks a list of tensors.

    :param x:     [list]   A list of TensorFlow tensor objects.

    :return:      [Tensor] Stacked tensor.
    """
    if tf.__version__.startswith('0'):    # 0.12 compatibility.
        return tf.pack(x)
    else:
        return tf.stack(x)


def cnn(x,
        filter_size,
        strides,
        pool_fn,
        pool_size,
        pool_strides,
        act_fn,
        dtype=tf.float32,
        add_bias=True,
        weight_decay=None,
        scope='cnn',
        trainable=True):
    """
    Builds a convolutional neural networks.
    Each layer contains the following operations:
        1) Convolution, y = w * x.
        2) Additive bias (optional), y = w * x + b.
        3) Activation function (optional), y = g( w * x + b ).
        4) Pooling (optional).

    Layers are having the following naming convention, e.g.
        - cnn/layer_0/w       # Conv filters
        - cnn/layer_0/b       # Conv filter biases
        - cnn/layer_0/act:0   # Output activation
        - cnn/layer_0/pool:0  # Output pooled activation
    Use graph.get_tensor_by_name to query for activation from a specific layer.

    :param x:            [Tensor]    Input variable.
    :param filter_size:  [list]      Shape of the convolutional filters, list of 4-d int.
    :param strides:      [list]      Convolution strides, list of 4-d int.
    :param pool_fn:      [list]      Pooling functions, list of N callable objects.
    :param pool_size:    [list]      Pooling field size, list of 4-d int.
    :param pool_strides: [list]      Pooling strides, list of 4-d int.
    :param act_fn:       [list]      Activation functions, list of N callable objects.
    :param dtype         [dtype]     Data type, one of TensorFlow dtypes.
    :param add_bias:     [bool]      Whether adding bias or not, bool.
    :param weight_decay:           [float]     Weight decay, float.
    :param scope:        [string]    Scope of the model, str.
    :param trainable     [bool]      Whether the weights are trainable.

    :return              [Tensor]    Last activation of the CNN.
    """
    num_layer = len(filter_size)
    h = x
    with tf.variable_scope(scope):
        for ii in range(num_layer):
            with tf.variable_scope('layer_{}'.format(ii)):
                w = weight_variable_cpu(
                    filter_size[ii],
                    init_method='xavier',
                    dtype=dtype,
                    weight_decay=weight_decay,
                    name='w',
                    trainable=trainable)
                h = tf.nn.conv2d(h, w, strides=strides[ii], padding='SAME', name='conv')
                if add_bias:
                    b = weight_variable_cpu(
                        [filter_size[ii][3]],
                        init_method='constant',
                        dtype=dtype,
                        init_param={'val': 0},
                        name='b',
                        trainable=trainable)
                    h = tf.add(h, b, name='conv_bias')
                if act_fn[ii] is not None:
                    h = act_fn[ii](h, name='act')
                if pool_fn[ii] is not None:
                    h = pool_fn[ii](
                        h, pool_size[ii], strides=pool_strides[ii], padding='SAME', name='pool')
    return h


def mlp(x,
        dims,
        is_training=True,
        act_fn=None,
        dtype=tf.float32,
        add_bias=True,
        weight_decay=None,
        scope='mlp',
        dropout=None,
        trainable=True):
    """
    Builds a multi-layer perceptron.
    Each layer contains the following operations:
        1) Linear transformation, y = w^T x.
        2) Additive bias (optional), y = w^T x + b.
        3) Activation function (optional), y = g( w^T x + b )
        4) Dropout (optional)

    :param x:            [Tensor]    Input variable.
    :param dims:         [list]      Layer dimensions, list of N+1 int.
    :param is_training   [bool]      Whether is in training mode.
    :param act_fn:       [list]      Activation functions, list of N callable objects.
    :param add_bias:     [bool]      Whether adding bias or not.
    :param weight_decay:           [float]     Weight decay.
    :param scope:        [string]    Scope of the model.
    :param dropout:      [list]      Whether to apply dropout, None or list of N bool.

    :return              [Tensor]    Last activation of the MLP.
    """
    num_layer = len(dims) - 1
    h = x
    with tf.variable_scope(scope):
        for ii in range(num_layer):
            with tf.variable_scope('layer_{}'.format(ii)):
                dim_in = dims[ii]
                dim_out = dims[ii + 1]
                w = weight_variable_cpu(
                    [dim_in, dim_out],
                    init_method='xavier',
                    dtype=dtype,
                    weight_decay=weight_decay,
                    name='w',
                    trainable=trainable)
                h = tf.matmul(h, w, name='linear')
                if add_bias:
                    b = weight_variable_cpu(
                        [dim_out],
                        init_method='constant',
                        dtype=dtype,
                        init_param={'val': 0.0},
                        name='b',
                        trainable=trainable)
                    h = tf.add(h, b, name='linear_bias')
                if act_fn and act_fn[ii] is not None:
                    h = act_fn[ii](h)
                if dropout is not None and dropout[ii]:
                    log.info('Apply dropout 0.5')
                    if is_training:
                        keep_prob = 0.5
                    else:
                        keep_prob = 1.0
                    h = tf.nn.dropout(h, keep_prob=keep_prob)
    return h


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               eps=1e-10,
               name='bn_out',
               decay=0.99,
               dtype=tf.float32,
               data_format='NHWC'):
    """
    Applies batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
    x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    :param x:       [Tensor]    Input tensor.
    :param gamma:   [Tensor]    Scaling parameter.
    :param beta:    [Tensor]    Bias parameter.
    :param axes:    [list]      Axes to collect statistics.
    :param eps:     [float]     Denominator bias.

    :return normed: [Tensor]    Batch-normalized variable.
    :return ops:    [list]      List of EMA ops.
    """
    if data_format == 'NHWC':
        n_out = x.get_shape()[-1]
        axes = [0, 1, 2]
    elif data_format == 'NCHW':
        n_out = x.get_shape()[1]
        axes = [0, 2, 3]
    try:
        n_out = int(n_out)
        shape = [n_out]
    except:
        shape = None
    emean = tf.get_variable(
        'ema_mean',
        shape=shape,
        trainable=False,
        dtype=dtype,
        initializer=tf.constant_initializer(0.0, dtype=dtype))
    evar = tf.get_variable(
        'ema_var',
        shape=shape,
        trainable=False,
        dtype=dtype,
        initializer=tf.constant_initializer(1.0, dtype=dtype))
    if is_training:
        mean, var = tf.nn.moments(x, axes, name='moments')
        ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
        ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps, name=name)
        return normed, [ema_mean_op, ema_var_op]
    else:
        normed = tf.nn.batch_normalization(x, emean, evar, beta, gamma, eps, name=name)
        return normed, None
