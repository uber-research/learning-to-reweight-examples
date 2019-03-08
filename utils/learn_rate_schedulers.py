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
# Learning rate scheduler utilities
#
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import abstractmethod
from utils import logger

log = logger.get()


class LearnRateScheduler(object):
    @abstractmethod
    def step(self, niter):
        pass


class FixedLearnRateScheduler(LearnRateScheduler):
    """Adjusts learning rate according to a fixed schedule."""

    def __init__(self, sess, model, base_learn_rate, learn_rate_decay_steps, learn_rate_list):
        """
        Initializes fixed learning rate scheduler.

        :param sess:                     [Session]  TensorFlow session object.
        :param model:                    [object]   Model object.
        :param base_learn_rate:          [float]    Base learning rate.
        :param learn_rate_decay_steps:   [list]     A list of step number which we perform decay.
        :param learn_rate_list:          [list]     A list of learning rate decay multiplier.
        """
        self.model = model
        self.sess = sess
        self.learn_rate = base_learn_rate
        self.learn_rate_list = learn_rate_list
        self.learn_rate_decay_steps = learn_rate_decay_steps
        self.model.assign_learn_rate(self.sess, self.learn_rate)

    def step(self, niter):
        """
        Adds to counter. Adjusts learning rate if necessary.

        :param niter:            [int]      Current number of iterations.
        """
        if len(self.learn_rate_decay_steps) > 0:
            if (niter + 1) == self.learn_rate_decay_steps[0]:
                self.learn_rate = self.learn_rate_list[0]
                self.model.assign_learn_rate(self.sess, self.learn_rate)
                self.learn_rate_decay_steps.pop(0)
                self.learn_rate_list.pop(0)
            elif (niter + 1) > self.learn_rate_decay_steps[0]:
                ls = self.learn_rate_decay_steps
                while len(ls) > 0 and (niter + 1) > ls[0]:
                    ls.pop(0)
                    self.learn_rate = self.learn_rate_list.pop(0)
                self.model.assign_learn_rate(self.sess, self.learn_rate)
