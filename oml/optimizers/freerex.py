from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.optimizers.optimizer import Optimizer
from oml.models.components import ProximalOracle

import numpy as np


class FreeRex(Optimizer):
    """
    Ashok Cutkosky, Kwabena Boahen
    "Online Learning Without Prior Information"
    COLT2017
    """

    def __init__(self, model, step_size=0.01, t=0, k=1):
        Optimizer.__init__(self, model, t=t)
        self.state['one_of_squared_eta'] = {}
        self.state['L_max'] = {}
        self.state['cumulative_grad'] = {}
        self.state['a'] = {}
        self.hyper_parameter['k'] = k

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        if isinstance(layer.param[key], ProximalOracle):
            grad += layer.param[key].reg.sub_differential(layer.param[key].param)

        self.state['cumulative_grad'][str(i) + key] \
            = grad + self.state['cumulative_grad'].get(str(i) + key, np.zeros_like(grad))
        self.state['L_max'][str(i) + key] = max(
            np.linalg.norm(grad),
            self.state['L_max'].get(str(i) + key, 0)
        )
        self.state['one_of_squared_eta'][str(i) + key] = max(
            self.state['one_of_squared_eta'].get(str(i) + key, 0) + 2 * np.linalg.norm(grad) ** 2,
            self.state['L_max'][str(i) + key] * np.linalg.norm(self.state['cumulative_grad'][str(i) + key])
        )
        self.state['a'][str(i) + key] = max(
            self.state['a'].get(str(i) + key, 0),
            self.state['one_of_squared_eta'][str(i) + key] / (self.state['L_max'][str(i) + key] ** 2)
        )
        layer.param[key].param \
            = -self.state['cumulative_grad'][str(i) + key] / (
            self.state['a'][str(i) + key] * np.linalg.norm(self.state['cumulative_grad'][str(i) + key])
        ) * (
                  np.exp(
                      np.linalg.norm(self.state['cumulative_grad'][str(i) + key]) / (
                          np.sqrt(self.state['one_of_squared_eta'][str(i) + key]) * self.hyper_parameter['k']
                      )
                  ) - 1
              )
