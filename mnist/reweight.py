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
# Models for MNIST experiments.
#
from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def get_model(inputs,
              labels,
              is_training=True,
              dtype=tf.float32,
              w_dict=None,
              ex_wts=None,
              reuse=None):
    """Builds a simple LeNet.
    
    :param inputs:            [Tensor]    Inputs.
    :param labels:            [Tensor]    Labels.
    :param is_training:       [bool]      Whether in training mode, default True.
    :param dtype:             [dtype]     Data type, default tf.float32.
    :param w_dict:            [dict]      Dictionary of weights, default None.
    :param ex_wts:            [Tensor]    Example weights placeholder, default None.
    :param reuse:             [bool]      Whether to reuse variables, default None.
    """

    if w_dict is None:
        w_dict = {}

    def _get_var(name, shape, dtype, initializer):
        key = tf.get_variable_scope().name + '/' + name
        if key in w_dict:
            return w_dict[key]
        else:
            var = tf.get_variable(name, shape, dtype, initializer=initializer)
            w_dict[key] = var
            return var

    with tf.variable_scope('Model', reuse=reuse):
        inputs_ = tf.cast(tf.reshape(inputs, [-1, 28, 28, 1]), dtype)
        labels = tf.cast(labels, dtype)

        w_init = tf.truncated_normal_initializer(stddev=0.1)
        w1 = _get_var('w1', [5, 5, 1, 16], dtype, initializer=w_init)    # [14, 14, 16]
        w2 = _get_var('w2', [5, 5, 16, 32], dtype, initializer=w_init)    # [7, 7, 32]
        w3 = _get_var('w3', [5, 5, 32, 64], dtype, initializer=w_init)    # [4, 4, 64]
        w4 = _get_var('w4', [1024, 100], dtype, initializer=w_init)
        w5 = _get_var('w5', [100, 1], dtype, initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b1 = _get_var('b1', [16], dtype, initializer=b_init)
        b2 = _get_var('b2', [32], dtype, initializer=b_init)
        b3 = _get_var('b3', [64], dtype, initializer=b_init)
        b4 = _get_var('b4', [100], dtype, initializer=b_init)
        b5 = _get_var('b5', [1], dtype, initializer=b_init)

        act = tf.nn.relu

        # Conv-1
        l0 = tf.identity(inputs_, name='l0')
        z1 = tf.add(tf.nn.conv2d(inputs_, w1, [1, 1, 1, 1], 'SAME'), b1, name='z1')
        l1 = act(tf.nn.max_pool(z1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l1')

        # Conv-2
        z2 = tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2, name='z2')
        l2 = act(tf.nn.max_pool(z2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l2')

        # Conv-3
        z3 = tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3, name='z3')
        l3 = act(tf.nn.max_pool(z3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l3')

        # FC-4
        z4 = tf.add(tf.matmul(tf.reshape(l3, [-1, 1024]), w4), b4, name='z4')
        l4 = act(z4, name='l4')

        # FC-5
        z5 = tf.add(tf.matmul(l4, w5), b5, name='z5')

        logits = tf.squeeze(z5)
        out = tf.sigmoid(logits)
        if ex_wts is None:
            # Average loss.
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            # Weighted loss.
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) * ex_wts)
    return w_dict, loss, logits


def reweight_random(bsize, eps=0.0):
    """Reweight examples using random numbers.
    
    :param bsize:             [int]       Batch size.
    :param eps:               [float]     Minimum example weights, default 0.0.
    """
    ex_weight = tf.random_normal([bsize], mean=0.0, stddev=1.0)
    ex_weight_plus = tf.maximum(ex_weight, eps)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.to_float(tf.equal(ex_weight_sum, 0.0))
    ex_weight_norm = ex_weight_plus / ex_weight_sum
    return ex_weight_norm


def reweight_autodiff(inp_a,
                      label_a,
                      inp_b,
                      label_b,
                      bsize_a,
                      bsize_b,
                      eps=0.0,
                      gate_gradients=1):
    """Reweight examples using automatic differentiation.

    :param inp_a:             [Tensor]    Inputs for the noisy pass.
    :param label_a:           [Tensor]    Labels for the noisy pass.
    :param inp_b:             [Tensor]    Inputs for the clean pass.
    :param label_b:           [Tensor]    Labels for the clean pass.
    :param bsize_a:           [int]       Batch size for the noisy pass.
    :param bsize_b:           [int]       Batch size for the clean pass.
    :param eps:               [float]     Minimum example weights, default 0.0.
    :param gate_gradients:    [int]       Tensorflow gate gradients, reduce concurrency.
    """
    ex_wts_a = tf.zeros([bsize_a], dtype=tf.float32)
    ex_wts_b = tf.ones([bsize_b], dtype=tf.float32) / float(bsize_b)
    w_dict, loss_a, logits_a = get_model(
        inp_a, label_a, ex_wts=ex_wts_a, is_training=True, reuse=True)
    var_names = w_dict.keys()
    var_list = [w_dict[kk] for kk in var_names]
    grads = tf.gradients(loss_a, var_list, gate_gradients=gate_gradients)

    var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
    w_dict_new = dict(zip(var_names, var_list_new))
    _, loss_b, logits_b = get_model(
        inp_b, label_b, ex_wts=ex_wts_b, is_training=True, reuse=True, w_dict=w_dict_new)
    grads_ex_wts = tf.gradients(loss_b, [ex_wts_a], gate_gradients=gate_gradients)[0]
    ex_weight = -grads_ex_wts
    ex_weight_plus = tf.maximum(ex_weight, eps)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.to_float(tf.equal(ex_weight_sum, 0.0))
    ex_weight_norm = ex_weight_plus / ex_weight_sum
    return ex_weight_norm


def reweight_hard_mining(inp, label, positive=False):
    """Reweight examples using hard mining.
    
    :param inp:          [Tensor]   [N, ...] Inputs.
    :param label:        [Tensor]   [N] Labels
    :param positive:     [bool]     Whether perform hard positive mining or hard negative mining.

    :return              [Tensor]   Examples weights of the same shape as the first dim of inp.
    """
    _, loss, logits = get_model(inp, label, ex_wts=None, is_training=True, reuse=True)

    # Mine for positive
    if positive:
        loss_mask = loss * label
    else:
        loss_mask = loss * (1 - label)

    if positive:
        k = tf.cast(tf.reduce_sum(1 - label), tf.int32)
    else:
        k = tf.cast(tf.reduce_sum(label), tf.int32)
    k = tf.maximum(k, 1)
    loss_sorted, loss_sort_idx = tf.nn.top_k(loss_mask, k)

    if positive:
        mask = 1 - label
    else:
        mask = label

    updates = tf.ones([tf.shape(loss_sort_idx)[0]], dtype=label.dtype)
    mask_add = tf.scatter_nd(tf.expand_dims(loss_sort_idx, axis=1), updates, [tf.shape(inp)[0]])
    mask = tf.maximum(mask, mask_add)
    mask_sum = tf.reduce_sum(mask)
    mask_sum += tf.cast(tf.equal(mask_sum, 0.0), tf.float32)
    mask = mask / mask_sum
    return mask
