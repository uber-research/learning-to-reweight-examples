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
# Data factory.
#
from __future__ import absolute_import, division, print_function, unicode_literals

from base.factory import Factory

_data_factory = Factory()
_inp_factory = Factory()


def RegisterDataset(dataset_name):
    """Registers a configuration."""
    return _data_factory.register(dataset_name)


def RegisterInputPipeline(input_pipeline_name):
    """Registers an input pipeline."""
    return _inp_factory.register(input_pipeline_name)


def get_dataset_cls(dataset_name):
    return _data_factory.get(dataset_name)


def get_input_pipeline_cls(input_pipeline_name):
    return _inp_factory.get(input_pipeline_name)


def get_data_inputs(dataset_name, data_dir, subset, is_training, batch_size, input_pipeline_name,
                    **kwargs):
    """
    Gets a data input instance.

    :param dataset_name:           [string]      Name of the dataset.
    :param subset:                 [string]      Subset of the dataset, train or validation.
    :param is_training:            [bool]        Whether in training mode.
    :param batch_size              [int]         Size of a mini-batch.
    :param input_pipeline_name:    [string]      Name of the input pipeline.

    :return                        [object]      InputPipeline object.
    """
    dataset = _data_factory.create(dataset_name, data_dir=data_dir, subset=subset, **kwargs)
    inp = _inp_factory.create(
        input_pipeline_name, dataset, is_training=is_training, batch_size=batch_size)
    return inp
