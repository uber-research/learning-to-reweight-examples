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
# Tests automatic differentiation.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import six
import tensorflow as tf

from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig
from models.cnn.fake_resnet_model import FakeResnetModel    # NOQA
from models.model_factory import get_model
from models.reweight_model import reweight_autodiff
from utils import logger, gen_id

from collections import namedtuple
from google.protobuf.text_format import Merge, MessageToString

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)

log = logger.get()


def _get_config():
    """Gets config."""
    config_file = 'cifar/configs/resnet-test.prototxt'
    config = ResnetModelConfig()
    Merge(open(config_file).read(), config)
    return config


def _get_model(config, inp, label, bsize, is_training, name_scope, reuse):
    """Builds models."""
    model_cls = 'fake-resnet'

    trn_kwargs = {
        'is_training': is_training,
        'inp': inp,
        'label': label,
        'batch_size': bsize,
    }
    with tf.name_scope(name_scope):
        with tf.variable_scope('Model', reuse=reuse):
            m = get_model(model_cls, config, **trn_kwargs)
    return m


def _get_assign_weighted_model(config, inp, label, weights, weights_dict, bsize, is_training,
                               name_scope, reuse):
    """Builds models."""
    model_cls = 'fake-assign-wts-resnet'

    trn_kwargs = {
        'is_training': is_training,
        'inp': inp,
        'label': label,
        'ex_wts': weights,
        'batch_size': bsize
    }
    with tf.name_scope(name_scope):
        with tf.variable_scope('Model', reuse=reuse):
            m = get_model(model_cls, config, weights_dict, **trn_kwargs)
    return m


def get_ex_weights_auto_diff(sess, model_a, model_b, ex_weights, data_a, data_b):
    # Get data.
    inp_a, label_a = sess.run([data_a.inputs, data_a.labels])
    dict_a = model_a._get_feed_dict(inp=inp_a, label=label_a)
    inp_b, label_b = sess.run([data_b.inputs, data_b.labels])
    dict_b = model_b._get_feed_dict(inp=inp_b, label=label_b)

    fdict = {}
    for kk in dict_a.keys():
        fdict[kk] = dict_a[kk]
    for kk in dict_b.keys():
        fdict[kk] = dict_b[kk]
    ex_weights_value = sess.run(ex_weights, feed_dict=fdict)
    return ex_weights_value


def get_ex_weights_finite_diff(sess, model_a, model_b, ex_wts_a, data_a, data_b):
    inp_a, label_a = sess.run([data_a.inputs, data_a.labels])
    dict_a = model_a._get_feed_dict(inp=inp_a, label=label_a)
    inp_b, label_b = sess.run([data_b.inputs, data_b.labels])
    dict_b = model_b._get_feed_dict(inp=inp_b, label=label_b)
    fdict = {}
    for kk in dict_a.keys():
        fdict[kk] = dict_a[kk]
    for kk in dict_b.keys():
        fdict[kk] = dict_b[kk]
    bsize = inp_a.shape[0]
    eps = 1e-5
    grads = np.zeros([bsize])

    for ii in six.moves.xrange(bsize):
        ex_wts_ = np.zeros([bsize])
        ex_wts_[ii] = eps
        fdict[ex_wts_a] = ex_wts_
        loss_plus = sess.run(model_b.cost, feed_dict=fdict)

        ex_wts_[ii] = -eps
        fdict[ex_wts_a] = ex_wts_
        loss_minus = sess.run(model_b.cost, feed_dict=fdict)
        grads[ii] = (loss_plus - loss_minus) / 2.0 / eps

    grads = np.maximum(-grads, 0.0)
    ex_weights_value = grads / grads.sum()
    return ex_weights_value


def main():
    # -----------------------------------------------------------------
    # Loads parammeters.
    config = _get_config()

    bsize_a = 10
    bsize_b = 10

    log.info('Config: {}'.format(MessageToString(config)))

    # Initializes variables.
    np.random.seed(0)
    if not hasattr(config, 'seed'):
        tf.set_random_seed(1234)
        log.info('Setting tensorflow random seed={:d}'.format(1234))
    else:
        log.info('Setting tensorflow random seed={:d}'.format(config.seed))
        tf.set_random_seed(config.seed)

    with tf.Graph().as_default(), tf.Session() as sess, tf.device('/cpu:0'):
        # -----------------------------------------------------------------
        # Building datasets.
        inp_a = np.random.uniform(-1.0, 1.0, [bsize_a, 32, 32, 3])
        label_a = np.floor(np.random.uniform(0.0, 10.0, [bsize_a])).astype(np.int64)
        inp_b = np.random.uniform(-1.0, 1.0, [bsize_a, 32, 32, 3])
        label_b = np.floor(np.random.uniform(0.0, 10.0, [bsize_a])).astype(np.int64)
        Dataset = namedtuple('Dataset', ['inputs', 'labels'])
        data_a = Dataset(inputs=tf.constant(inp_a), labels=tf.constant(label_a))
        data_b = Dataset(inputs=tf.constant(inp_b), labels=tf.constant(label_b))

        # -----------------------------------------------------------------
        # Build shared-weights model replicas, for different data sources.
        var_list = tf.trainable_variables()
        [print(vv.name) for vv in var_list]

        ex_wts_a = tf.placeholder_with_default(
            tf.zeros([bsize_a], dtype=tf.float64), [bsize_a], name='ex_wts_a')
        model_c = _get_model(config, None, None, bsize_a, True, 'TrainC', None)

        def build_model_a(wts_dict, ex_wts, reuse):
            return _get_assign_weighted_model(config, None, None, ex_wts, wts_dict, bsize_a, True,
                                              'TrainA', reuse)

        def build_model_b(wts_dict, ex_wts, reuse):
            return _get_assign_weighted_model(config, None, None, ex_wts, wts_dict, bsize_b, True,
                                              'TrainB', reuse)

        model_a, model_b, ex_weights = reweight_autodiff(
            build_model_a,
            build_model_b,
            config.optimizer_config.learn_rate,
            bsize_a,
            bsize_b,
            ex_wts_a=ex_wts_a,
            gate_gradients=2,
            legacy=True)
        sess.run(tf.global_variables_initializer())
        ex_ad = get_ex_weights_auto_diff(sess, model_a, model_b, ex_weights, data_a, data_b)
        ex_fd = get_ex_weights_finite_diff(sess, model_a, model_b, ex_wts_a, data_a, data_b)
        log.info('Auto diff: {}'.format(ex_ad))
        log.info('Finite diff: {}'.format(ex_fd))
        np.testing.assert_allclose(ex_ad, ex_fd)
        log.info('pass')


if __name__ == '__main__':
    flags = tf.flags
    FLAGS = flags.FLAGS
    log.setLevel(logging.DEBUG)
    main()
