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
# Model factory for creating new model instances.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import json

from collections import namedtuple
from base.factory import Factory
from utils import logger

log = logger.get()

_factory = Factory()


def RegisterModel(model_name):
    """Registers a configuration."""
    return _factory.register(model_name)


def get_model(model_name, config, *args, **kwargs):
    """
    Gets a model instance from predefined library.

    :param model_name: String. Name of the model.
    :param config: Configuration object.

    :return: A Model instance.
    """
    config_copy = type(config)()
    config_copy.ParseFromString(config.SerializeToString())
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
        if batch_size is not None:
            log.info("Batch size is set to {}".format(batch_size))

    if not _factory.has(model_name):
        raise ValueError("Unknown model \"{}\"".format(model_name))

    def _get_model(*args, **kwargs):
        return _factory.create(model_name, *args, **kwargs)

    return _get_model(config_copy, *args, **kwargs)


def get_model_from_file(model_name, config_file, *args, **kwargs):
    if not _factory.has(model_name):
        raise ValueError("Unknown model \"{}\"".format(model_name))
    return _factory.get(model_name).create_from_file(config_file, *args, **kwargs)
