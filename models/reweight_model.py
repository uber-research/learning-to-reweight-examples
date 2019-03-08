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
# Reweighting algorithms.
#
import tensorflow as tf


def _reweight(build_model_a,
              build_model_b,
              bsize_a,
              bsize_b,
              ex_wts_a=None,
              gate_gradients=1,
              legacy=False):
    """Reweight function.

    :param build_model_a:     [function]    Function that can build the model graph for the noisy pass.
    :param build_model_b:     [function]    Function that can build the model graph for the clean pass.
    :param bsize_a:           [int]         Batch size for the noisy pass.
    :param bsize_b:           [int]         Batch size for the clean pass.
    :param ex_wts_a:          [Tensor]      Initial weights, default 0.
    :param gate_gradients:    [int]         Gate gradient status, more accurate gradient by reduce concurrency.
    :param legacy:            [bool]        Use legacy mode, which is slower but allows gradient check.

    :return                   [object]      Model for the noisy pass.
    :return                   [object]      Model for the clean pass.
    :return                   [Tensor]      Negative gradients of the example weights that are of shape [bsize_a].
    """
    if ex_wts_a is None:
        ex_wts_a = tf.zeros([bsize_a], dtype=tf.float32)
    ex_wts_b = tf.ones([bsize_b], dtype=ex_wts_a.dtype) / float(bsize_b)
    model_a = build_model_a(None, ex_wts_a, True)
    var_names = model_a._cnn_module.weights_dict.keys()
    var_list = [model_a._cnn_module.weights_dict[kk] for kk in var_names]
    grads_a = tf.gradients(model_a.cost, var_list, gate_gradients=gate_gradients)
    if legacy:    # For finite difference gradient checking only, slower.
        grad_dict = dict(zip(var_names, grads_a))
        weights_dict_new = dict()
        for kk in model_a._cnn_module.weights_dict.keys():
            weights_dict_new[kk] = model_a._cnn_module.weights_dict[kk] - grad_dict[kk]
    else:
        weights_dict_new = model_a._cnn_module.weights_dict
    model_b = build_model_b(weights_dict_new, ex_wts_b, True)
    if legacy:
        grads_ex_wts = -tf.gradients(model_b.cost, [ex_wts_a], gate_gradients=gate_gradients)[0]
    else:
        grads_b = tf.gradients(model_b.cost, var_list, gate_gradients=gate_gradients)
        grads_ex_wts = tf.gradients(grads_a, [ex_wts_a], grads_b, gate_gradients=gate_gradients)[0]
    return model_a, model_b, grads_ex_wts


def reweight_autodiff(build_model_a,
                      build_model_b,
                      bsize_a,
                      bsize_b,
                      ex_wts_a=None,
                      eps=0.0,
                      gate_gradients=1,
                      legacy=False):
    """Reweight model using autodiff.

    :param build_model_a:     [function]    Function that can build the model graph for the noisy pass.
    :param build_model_b:     [function]    Function that can build the model graph for the clean pass.
    :param bsize_a:           [int]         Batch size for the noisy pass.
    :param bsize_b:           [int]         Batch size for the clean pass.
    :param ex_wts_a:          [Tensor]      Initial weights, default 0.
    :param gate_gradients:    [int]         Gate gradient status, more accurate gradient by reduce concurrency.
    :param legacy:            [bool]        Use legacy mode, which is slower but allows gradient check.

    :return                   [object]      Model for the noisy pass.
    :return                   [object]      Model for the clean pass.
    :return                   [Tensor]      Example weights that are of shape [bsize_a].
    """
    model_a, model_b, grads_ex_wts = _reweight(
        build_model_a,
        build_model_b,
        bsize_a,
        bsize_b,
        ex_wts_a=ex_wts_a,
        gate_gradients=gate_gradients,
        legacy=legacy)
    ex_weight_plus = tf.maximum(grads_ex_wts, eps)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.cast(tf.equal(ex_weight_sum, 0.0), ex_weight_sum.dtype)
    ex_weight_norm = ex_weight_plus / ex_weight_sum
    # Do not take gradients of the example weights again.
    ex_weight_norm = tf.stop_gradient(ex_weight_norm)
    return model_a, model_b, ex_weight_norm
