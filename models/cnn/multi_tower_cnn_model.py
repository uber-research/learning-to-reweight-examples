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
# Multi-tower CNN model for training CNN on multiple towers (multiple GPUs).
#
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from models.cnn.cnn_model import CNNModel
from utils import logger

log = logger.get()


class MultiTowerCNNModel(CNNModel):
    """Multi Tower CNN Model."""

    def __init__(self,
                 config,
                 cnn_module,
                 is_training=True,
                 inp=None,
                 label=None,
                 batch_size=None,
                 num_replica=2):
        """
        Multi Tower CNN constructor.

        :param config:      [object]    Configuration object.
        :param cnn_module:  [object]    A CNN module which builds the main graph.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        :param num_replica: [int]       Number of in-graph replicas (number of GPUs).
        """
        with tf.device(self._get_device("cpu", 0)):
            self._num_replica = num_replica
            self._avg_cost = None
            self._avg_cross_ent = None
            self._stack_output = None
            super(MultiTowerCNNModel, self).__init__(
                config,
                cnn_module,
                is_training=is_training,
                inp=inp,
                label=label,
                batch_size=batch_size)

    def _build_graph(self, inp):
        """
        Builds core computation graph from inputs to outputs.

        :param inp:            [Tensor]     4D float tensor, inputs to the network.

        :return                [Tensor]     output tensor.
        """
        inputs = tf.split(inp, self.num_replica, axis=0)
        outputs = []
        for ii in range(self.num_replica):
            _device = self._get_replica_device(ii)
            with tf.device(_device):
                with tf.name_scope("%s_%d" % ("replica", ii)):
                    outputs.append(self._cnn_module(inputs[ii]))
                    log.info("Replica {} forward built on {}".format(ii, _device))
                    tf.get_variable_scope().reuse_variables()
        # Reset reuse flag.
        tf.get_variable_scope()._reuse = None
        return outputs

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [list]      Outputs of the network from each tower.

        :return                 [list]      Loss value from each tower.
        """
        labels = tf.split(self.label, self.num_replica, axis=0)
        xent_list = []
        cost_list = []
        for ii, (_output, _label) in enumerate(zip(output, labels)):
            with tf.device(self._get_replica_device(ii)):
                with tf.name_scope("%s_%d" % ("replica", ii)):
                    _xent = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=_output, labels=_label))
                    xent_list.append(_xent)
                    _cost = _xent
                    if self.is_training:
                        _cost += self._decay()
                    cost_list.append(_cost)
        self._cross_ent = xent_list
        return cost_list

    def _compute_gradients(self, cost, var_list=None):
        """
        :params cost            [list]      List of loss values from each tower.

        :return                 [list]      List of pairs of (gradient, variable) where the gradient
        has been averaged across all towers.
        """
        grads_and_vars = []
        for ii, _cost in enumerate(cost):
            with tf.device(self._get_replica_device(ii)):
                with tf.name_scope("%s_%d" % ("replica", ii)):
                    var_list = tf.trainable_variables()
                    grads = tf.gradients(_cost, var_list)
                    grads_and_vars.append(list(zip(grads, var_list)))
        avg_grads_and_vars = self._average_gradients(grads_and_vars)
        return avg_grads_and_vars

    def _average_gradients(self, tower_grads):
        """
        Calculates the average gradient for each shared variable across all towers. Note that this
        function procides a synchronization point across all towers.

        :param tower_grads      [list]      List of lists of (gradient, variable) tuples. The inner
                                            list is over individual gradients. The outer list is
                                            over the gradient calculation for each tower.

        :return                 [list]      List of pairs of (gradient, variable) where the gradient
        has been averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g is None:
                    log.warning("No gradient for variable \"{}\"".format(v.name))
                    grads.append(None)
                    break
                else:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)

            # Average over the "tower" dimension.
            if grads[0] is None:
                grad = None
            else:
                grad = tf.concat(grads, axis=0)
                grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower"s pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _get_device(self, device_name="cpu", device_id=0):
        return "/{}:{:d}".format(device_name, device_id)

    def _get_replica_device(self, replica_id):
        device = self._get_device("gpu", replica_id)
        return device

    @property
    def cost(self):
        if self._avg_cost is None:
            self._avg_cost = tf.reduce_mean(tf.stack(self._cost))
        return self._avg_cost

    @property
    def cross_ent(self):
        if self._avg_cross_ent is None:
            self._avg_cross_ent = tf.reduce_mean(tf.stack(self._cross_ent))
        return self._avg_cross_ent

    @property
    def output(self):
        if self._stack_output is None:
            self._stack_output = tf.concat(self._output, axis=0)
        return self._stack_output

    @property
    def num_replica(self):
        return self._num_replica
