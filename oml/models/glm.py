from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from . import components
from oml.model import OracleRef, FirstOrderOracle

"""
Generalized Linear Models
"""


class BaseGLM(OracleRef):

    def __init__(self, layers, last_layer):
        self.layers = layers
        self.last_layer = last_layer

    def get_oracle(self):
        return [layer for layer in self.last_layer if isinstance(layer, FirstOrderOracle)]

    def predict(self, x):
        raise NotImplementedError()

    def loss(self, x, t):
        raise NotImplementedError()

    def fit(self, iterator, optimizer):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()


class LinearRegression(BaseGLM):

    def __init__(
            self,
            input_size,
            output_size,
    ):
        BaseGLM.__init__(
            self,
            [
                components.Affine(input_size, output_size)
            ],
            components.Gauss(),
        )

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.predict(x)

    def loss(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.forward(x, t)

    def fit(self, iterator, optimizer):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
