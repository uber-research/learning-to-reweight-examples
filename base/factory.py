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
# Factory implementation.
#
from __future__ import absolute_import, division, print_function, unicode_literals


class Factory(object):
    """Factory implementation."""

    def __init__(self):
        self._registry = dict()

    def register(self, name):
        """
        Registers a class.

        :param name: [string]   Name of the class.

        :return:     [function] A function that registers the class.
        """

        def decorator(f):
            self._registry[name] = f
            return f

        return decorator

    def has(self, name):
        """
        Checks if a name has been registered.

        :param name: [string]   Name of the class.

        :return:     [bool]     Whether the class has been registered.
        """
        return name in self._registry

    def create(self, name, *args, **kwargs):
        """
        Creates a class.

        :param name:   [string]   Name of the class.
        :param args:   [list]     Additional positional arguments.
        :param kwargs: [dict]     Additional named arguments.

        :return:       [object]   An instance of the class.
        """
        return self._registry[name](*args, **kwargs)

    def get(self, name):
        """Gets a class constructor.

        :param name:   [string]   Name of the class.

        :return:       [function] The class constructor.
        """
        return self._registry[name]
