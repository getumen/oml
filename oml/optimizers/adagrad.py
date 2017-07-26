from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.optimizers import optimizer
from oml.models.components import ProximalOracle
from oml.functions import StrongConvexity
import warnings


class AdaGrad(optimizer.Optimizer):
    """
    Duchi, John, Elad Hazan, and Yoram Singer.
    "Adaptive subgradient methods for online learning and stochastic optimization."
     Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            delta=1e-4,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.state['squared_cumulative_grad'] = {}

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        self.state['squared_cumulative_grad'][str(i) + key] \
            = self.state['squared_cumulative_grad'].get(str(i) + key, np.zeros_like(grad)) + np.multiply(grad, grad)

        layer.param[key].param -= \
            self.hyper_parameter['step_size'] * grad / (
                np.sqrt(self.state['squared_cumulative_grad'][str(i) + key])
                + self.hyper_parameter['delta']
            )

        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, self.hyper_parameter['step_size'] / (
                    np.sqrt(self.state['squared_cumulative_grad'][str(i) + key])
                    + self.hyper_parameter['delta']
                )
            )


class AdaRDA(optimizer.Optimizer):
    """
    Duchi, John, Elad Hazan, and Yoram Singer.
    "Adaptive subgradient methods for online learning and stochastic optimization."
     Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            delta=1e-4,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.state['averaged_cumulative_grad'] = {}
        self.state['squared_cumulative_grad'] = {}
        if not isinstance(model, StrongConvexity):
            warnings.warn("AdaRDA is not appropriate for optimizing non convex objective")

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        self.state['squared_cumulative_grad'][str(i) + key] = self.state['squared_cumulative_grad'].get(
            str(i) + key, np.zeros_like(grad)
        ) + np.multiply(grad, grad)
        self.state['averaged_cumulative_grad'][str(i) + key] = \
            (
                self.state['averaged_cumulative_grad'].get(str(i) + key, np.zeros_like(grad)) * (self.t - 1)
                + grad
            ) / self.t

        layer.param[key].param = \
            -self.t * self.hyper_parameter['step_size'] * self.state['averaged_cumulative_grad'][str(i) + key] / (
                np.sqrt(self.state['squared_cumulative_grad'][str(i) + key]) +
                self.hyper_parameter['delta']
            )

        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, self.hyper_parameter['step_size'] * self.t / (
                    np.sqrt(self.state['squared_cumulative_grad'][str(i) + key])
                    + self.hyper_parameter['delta']
                )
            )
