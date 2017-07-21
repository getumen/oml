from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals


"""
Generalized Linear Models
"""


class BaseGLM:

    def __init__(self, serializer=None):
        pass

    def predict(self, x):
        raise NotImplementedError()

    def loss(self, x, y):
        raise NotImplementedError()

    def fit(self, iterator, optimizer):
        raise NotImplementedError()
