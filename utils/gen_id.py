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
# Generates experiment ID.
#
import uuid


def get(name):
    """
    Returns a unique experiment ID.

    :param name:   [string]  Prefix of the experiment.
    """
    return name + '_' + str(uuid.uuid1())
