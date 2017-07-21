from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division


class BaseIterator:

    def __init__(self):
        self._start = False

    def __iter__(self):
        if self._start:
            raise ValueError('Iterator is already start')
        self._start = True
        return self

    def next(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

