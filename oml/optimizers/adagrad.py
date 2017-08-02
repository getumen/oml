from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import numpy as np

from oml.functions import StrongConvexity
from oml.models.components import ProximalOracle
from oml.optimizers import optimizer


class AdaGrad(optimizer.Optimizer):
    """
    Duchi, John, Elad Hazan, and Yoram Singer.
    "Adaptive subgradient methods for online learning and stochastic optimization."
     Journal of Machine Learning Research 12.Jul (2011): 2121-2159.

    For SC
    Mukkamala, Mahesh Chandra, and Matthias Hein.
    "Variants of RMSProp and Adagrad with Logarithmic Regret Bounds."
    arXiv preprint arXiv:1706.05507 (2017).
    ICML 2017( ´∀｀)ノ
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            delta=1e-8,
            xi1=1,
            xi2=1,
            exploit_sc=True,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.state['squared_cumulative_grad'] = {}
        self.hyper_parameter['exploit_sc'] = exploit_sc
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
            self.hyper_parameter['xi1'] = xi1
            self.hyper_parameter['xi2'] = xi2

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        self.state['squared_cumulative_grad'][str(i) + key] \
            = self.state['squared_cumulative_grad'].get(str(i) + key, np.zeros_like(grad)) + np.multiply(grad, grad)

        if self.hyper_parameter['mu'] > 0 and self.hyper_parameter['exploit_sc']:
            delta = self.hyper_parameter['xi2'] * np.exp(
                -self.hyper_parameter['xi1'] * self.state['squared_cumulative_grad'][str(i) + key]
            )
            a = delta + self.state['squared_cumulative_grad'][str(i) + key]
        else:
            delta = self.hyper_parameter['delta']
            a = delta + np.sqrt(self.state['squared_cumulative_grad'][str(i) + key])

        layer.param[key].param -= self.hyper_parameter['step_size'] * grad / a

        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, self.hyper_parameter['step_size'] / a
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
