from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.functions import Differentiable
from oml.models.components import FactorizationMachine, Gauss
from oml.models.model import Regression
from oml.models.regulizers import Nothing
from oml.models.components import State

import numpy as np


class FM(Regression, Differentiable):
    def __init__(
            self,
            input_bias_reg=Nothing(),
            variance_reg=Nothing()
    ):
        Regression.__init__(
            self,
            [FactorizationMachine(input_bias_reg=input_bias_reg, variance_reg=variance_reg)],
            Gauss(),
        )
        Differentiable.__init__(self, gamma=1)

    def predict(self, x: np.ndarray, *args, **kwargs):
        x_data = np.zeros(x.shape + (3,))

        for i in range(x.shape[0]):
            for j in range(2):
                x_data[i, j, 0] = j
                x_data[i, j, 1] = x[i, j]
                x_data[i, j, 2] = 1

        for layer in self.layers:
            x = layer.forward(x_data, *args, **kwargs)
        return self.last_layer.predict(x, *args, **kwargs)

    def loss(self, x: np.ndarray, t: np.ndarray, *args, **kwargs):
        reg = 0
        x_data = np.zeros(x.shape + (3,))

        for i in range(x.shape[0]):
            for j in range(2):
                x_data[i, j, 0] = j
                x_data[i, j, 1] = x[i, j]
                x_data[i, j, 2] = 1

        for layer in self.layers:
            x = layer.forward(x_data, *args, **kwargs)
            if isinstance(layer, State):
                for key in layer.param.keys():
                    reg += layer.param[key].reg.apply(layer.param[key].param)
        return self.last_layer.forward(x, t, *args, **kwargs) + reg
