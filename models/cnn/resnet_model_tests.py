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
# Unit tests for resnet_model.py.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import tensorflow as tf

from models.cnn.resnet_model import ResnetModel    # NOQA
from models.model_factory import get_model_from_file


class ResnetModelTests(tf.test.TestCase):
    """Tests the single for-loop implementation is the same as the double
    for-loop implementation."""

    def _test_getmodel(self, config_file):
        with tf.Graph().as_default(), self.test_session() as sess:
            np.random.seed(0)
            with tf.variable_scope('Model', reuse=None):
                m = get_model_from_file('resnet', config_file, is_training=True)
            config = m.config
            xval = np.random.uniform(
                0.0, 1.0, [10, config.input_height, config.input_width,
                           config.num_channels]).astype(np.float32)
            sess.run(tf.global_variables_initializer())
            m.infer_step(sess, inp=xval)

    def test_getmodel(self):
        """Tests initialize Resnet object."""
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'configs/resnet-test.prototxt')
        self._test_getmodel(filename)

    def test_getbtlmodel(self):
        """Tests initialize Resnet object with bottleneck."""
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'configs/resnet-test-btl.prototxt')
        self._test_getmodel(filename)


if __name__ == "__main__":
    tf.test.main()
