from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.models.components import ProximalOracle
from oml.optimizers import optimizer


class Adam(optimizer.Optimizer):
    """
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
            self,
            model,
            step_size=0.001,
            t=0,
            delta=1e-8,
            beta1=0.9,
            beta2=0.999
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.state['fst_moment'] = {}
        self.state['snd_moment'] = {}
        self.hyper_parameter['beta1'] = beta1
        self.hyper_parameter['beta2'] = beta2

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        if isinstance(layer.param[key], ProximalOracle):
            grad += layer.param[key].reg.sub_differential(layer.param[key].param)

        self.state['fst_moment'][str(i) + key] = self.state['fst_moment'].get(
            str(i) + key, np.zeros_like(grad)
        ) * self.hyper_parameter['beta1'] + grad * (1 - self.hyper_parameter['beta1'])

        self.state['snd_moment'][str(i) + key] = self.state['snd_moment'].get(
            str(i) + key, np.zeros_like(grad)
        ) * self.hyper_parameter['beta2'] + np.multiply(grad, grad) * (1 - self.hyper_parameter['beta2'])

        m = self.state['fst_moment'][str(i) + key] / (1 - self.hyper_parameter['beta1'] ** self.t)
        v = self.state['snd_moment'][str(i) + key] / (1 - self.hyper_parameter['beta2'] ** self.t)

        layer.param[key].param -= \
            self.hyper_parameter['step_size'] * m / (
                np.sqrt(np.sqrt(v)) + self.hyper_parameter['delta']
            )


class AdMax(optimizer.Optimizer):
    """
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
            self,
            model,
            step_size=0.001,
            t=0,
            delta=1e-8,
            beta1=0.9,
            beta2=0.999
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.hyper_parameter['delta'] = delta
        self.state['fst_moment'] = {}
        self.state['snd_moment'] = {}
        self.hyper_parameter['beta1'] = beta1
        self.hyper_parameter['beta2'] = beta2

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        if isinstance(layer.param[key], ProximalOracle):
            grad += layer.param[key].reg.sub_differential(layer.param[key].param)

        self.state['fst_moment'][str(i) + key] = self.state['fst_moment'].get(
            str(i) + key, np.zeros_like(grad)
        ) * self.hyper_parameter['beta1'] + grad * (1 - self.hyper_parameter['beta1'])

        self.state['snd_moment'][str(i) + key] = np.maximum(
            self.state['snd_moment'].get(str(i) + key, 0) * self.hyper_parameter['beta2'],
            np.absolute(grad)
        )

        layer.param[key].param -= \
            self.hyper_parameter['step_size'] / (1 - self.hyper_parameter['beta1'] ** self.t) \
            * self.state['fst_moment'][str(i) + key] / self.state['snd_moment'][str(i) + key]
