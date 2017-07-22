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

    def __next__(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()


class NumpyIterator(BaseIterator):

    def __init__(self, x, t):
        BaseIterator.__init__(self)
        self.x = x
        self.t = t
        self.pointer = 0

    def __next__(self):
        if self.pointer >= self.x.shape[0]:
            raise StopIteration()
        res = self.x[self.pointer, :].reshape(1, -1), self.t[self.pointer]
        self.pointer += 1
        return res

    def initialize(self):
        self.pointer = 0
        self._start = False
