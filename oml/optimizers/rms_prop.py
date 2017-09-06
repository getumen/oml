from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.functions import StrongConvexity
from oml.models.components import ProximalOracle
from . import optimizer


class RMSProp(optimizer.Optimizer):
    """
    Hinton
    in lecture

    RMS Prop has regret analysis now.
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
            beta=0.9,
            exploit_sc=True,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.hyper_parameter['beta'] = beta
        self.state['squared_cumulative_grad'] = {}
        self.hyper_parameter['exploit_sc'] = exploit_sc
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
            self.hyper_parameter['xi1'] = xi1
            self.hyper_parameter['xi2'] = xi2

    def rule(self, i, key, layer):
        grad = layer.param[key].grad

        if self.hyper_parameter['mu'] > 0 and self.hyper_parameter['exploit_sc']:
            beta = 1 - 1 / self.t
            self.state['squared_cumulative_grad'][str(i) + key] \
                = beta * self.state['squared_cumulative_grad'].get(
                str(i) + key, np.zeros_like(grad)
            ) + (1 - beta) * np.multiply(grad, grad)
            delta = self.hyper_parameter['xi2'] * np.exp(
                -self.hyper_parameter['xi1'] * self.t * self.state['squared_cumulative_grad'][str(i) + key]
            ) / self.t
            step_size = self.hyper_parameter['step_size'] / self.t
            a = delta + self.state['squared_cumulative_grad'][str(i) + key]
        else:
            self.state['squared_cumulative_grad'][str(i) + key] \
                = self.hyper_parameter['beta'] * self.state['squared_cumulative_grad'].get(
                str(i) + key, np.zeros_like(grad)
            ) + (1 - self.hyper_parameter['beta']) * np.multiply(grad, grad)
            delta = self.hyper_parameter['delta'] / np.sqrt(self.t)
            step_size = self.hyper_parameter['step_size'] / np.sqrt(self.t)
            a = delta + np.sqrt(self.state['squared_cumulative_grad'][str(i) + key])

        layer.param[key].param -= step_size * grad / a

        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, step_size / a
            )
