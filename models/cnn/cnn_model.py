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
# A general CNN model.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from models.optim.optimizer_factory import get_optimizer
from utils import logger

log = logger.get()


class CNNModel(object):
    """CNN model."""

    def __init__(self, config, cnn_module, is_training=True, inp=None, label=None, batch_size=None):
        """
        CNN constructor.

        :param config:      [object]    Configuration object.
        :param cnn_module:  [object]    A CNN module which builds the main graph.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        self._config = config
        self._bn_update_ops = None
        self._is_training = is_training
        self._batch_size = batch_size

        # Input.
        if inp is None:
            x = tf.placeholder(self.dtype, [
                batch_size, config.input_height, config.input_width, config.num_channels
            ], 'x')
        else:
            x = inp

        if label is None:
            y = tf.placeholder(tf.int32, [batch_size], 'y')
        else:
            y = label
        self._input = x
        self._label = y
        self._cnn_module = cnn_module

        logits = self._build_graph(x)
        cost = self._compute_loss(logits)
        self._cost = cost
        self._output = logits
        self._correct = self._compute_correct()

        if not is_training:
            return

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int64)
        learn_rate = tf.get_variable(
            'learn_rate', [],
            initializer=tf.constant_initializer(0.0),
            trainable=False,
            dtype=self.dtype)
        self._learn_rate = learn_rate
        self._grads_and_vars = self._compute_gradients(cost)
        log.info('BN update ops:')
        [log.info(op) for op in self.bn_update_ops]
        log.info('Total number of BN updates: {}'.format(len(self.bn_update_ops)))
        self._train_op = self._apply_gradients(
            self._grads_and_vars, global_step=global_step, name='train_step')
        self._global_step = global_step
        self._new_learn_rate = tf.placeholder(self.dtype, shape=[], name='new_learning_rate')
        self._learn_rate_update = tf.assign(self._learn_rate, self._new_learn_rate)

    def _build_graph(self, inp):
        """
        Builds core computation graph from inputs to outputs.

        :param inp:            [Tensor]     4D float tensor, inputs to the network.

        :return                [Tensor]     output tensor.
        """
        return self._cnn_module(inp)

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [Tensor]    Output of the network.

        :return                 [Scalar]    Loss value.
        """
        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.label)
            xent = tf.reduce_mean(xent, name='xent')
            cost = xent
            cost += self._decay()
            self._cross_ent = xent
        return cost

    def _compute_correct(self):
        """
        Computes number of correct predictions.

        :return                 [Scalar]    Number of correct predictions.
        """
        output_idx = tf.cast(tf.argmax(self.output, axis=1), self.label.dtype)
        return tf.to_float(tf.equal(output_idx, self.label))

    def assign_learn_rate(self, session, learn_rate_value):
        """
        Assigns new learning rate.

        :param session:          [Session]     TensorFlow Session object.
        :param learn_rate_value: [float]       New learning rate value.
        """
        log.info('Adjusting learning rate to {}'.format(learn_rate_value))
        session.run(self._learn_rate_update, feed_dict={self._new_learn_rate: learn_rate_value})

    def _apply_gradients(self, grads_and_vars, global_step=None, name='train_step'):
        """
        Applies the gradients globally.

        :param grads_and_vars: [list]      List of tuple of a gradient and a variable.
        :param global_step:    [Tensor]    Step number, optional.
        :param name:           [string]    Name of the operation, default 'train_step'.
        """
        # opt = get_optimizer(self.config.optimizer_config.type)(
        #     self.learn_rate, self.config.optimizer_config.momentum)
        opt = tf.train.MomentumOptimizer(self.learn_rate, 0.9)
        train_op = opt.apply_gradients(self._grads_and_vars, global_step=global_step, name=name)
        return train_op

    def _compute_gradients(self, cost, var_list=None):
        """
        Computes the gradients to variables.

        :param cost     [Tensor]     Cost function value.
        :param var_list [list]       List of variables to optimize, optional.

        :return:        [list]       List of tuple of a gradient and a variable.
        """
        if var_list is None:
            var_list = tf.trainable_variables()
        grads = tf.gradients(cost, var_list)
        return zip(grads, var_list)

    def _decay(self):
        """
        Applies L2 weight decay loss.

        :return: [Tensor]  Regularization cost function value.
        """
        weight_decay_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        log.info('Weight decay variables')
        [log.info(x) for x in weight_decay_losses]
        log.info('Total length: {}'.format(len(weight_decay_losses)))
        if len(weight_decay_losses) > 0:
            return tf.add_n(weight_decay_losses)
        else:
            log.warning('No weight decay variables!')
            return 0.0

    def _get_feed_dict(self, inp=None, label=None):
        """
        Generates feed dict.

        :param inp:      [ndarray]    Optional inputs, for placeholder only.
        :param label:    [ndarray]    Optional labels, for placeholder only.

        :return:         [dict]       A dictionary from model Tensor to input numpy arrays.
        """
        if inp is None and label is None:
            return None
        feed_data = {}
        if inp is not None:
            feed_data[self.input] = inp
        if label is not None:
            feed_data[self.label] = label
        return feed_data

    def infer_step(self, sess, inp=None):
        """
        Runs one inference step.

        :param sess:     [Session]    TensorFlow Session object.
        :param inp:      [ndarray]    Optional inputs, for placeholder only.

        :return:         [ndarray]    Output prediction.
        """
        return sess.run(self.output, feed_dict=self._get_feed_dict(inp=inp))

    def eval_step(self, sess, inp=None, label=None):
        """
        Runs one training step.

        :param sess:     [Session]    TensorFlow Session object.
        :param inp:      [ndarray]    Optional inputs, for placeholder only.
        :param label:    [ndarray]    Optional labels, for placeholder only.

        :return:         [float]      Accuracy value.
        :return:         [float]      Cross entropy value.
        """
        return sess.run(
            [self.correct, self.cross_ent], feed_dict=self._get_feed_dict(inp=inp, label=label))

    def train_step(self, sess, inp=None, label=None):
        """
        Runs one training step.

        :param sess:     [Session]    TensorFlow Session object.
        :param inp:      [ndarray]    Optional inputs, for placeholder only.
        :param label:    [ndarray]    Optional labels, for placeholder only.

        :return:         [float]      Cross entropy value.
        """
        results = sess.run(
            [self.cross_ent, self.train_op] + self.bn_update_ops,
            feed_dict=self._get_feed_dict(inp=inp, label=label))
        return results[0]

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def bn_update_ops(self):
        if self._bn_update_ops is None:
            self._bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return self._bn_update_ops

    @property
    def config(self):
        return self._config

    @property
    def learn_rate(self):
        return self._learn_rate

    @property
    def dtype(self):
        return tf.float32

    @property
    def is_training(self):
        return self._is_training

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def correct(self):
        if self._correct is None:
            self._correct = self._compute_correct(self.output)
        return self._correct

    @property
    def label(self):
        return self._label

    @property
    def cross_ent(self):
        return self._cross_ent

    @property
    def global_step(self):
        return self._global_step

    @property
    def grads_and_vars(self):
        return self._grads_and_vars
