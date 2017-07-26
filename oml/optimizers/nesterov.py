from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.functions import StrongConvexity, Differentiable
from oml.models.components import State, ProximalOracle
from oml.optimizers.optimizer import Optimizer


class AccSGD(Optimizer):
    """
    Hu, Chonghai, Weike Pan, and James T. Kwok.
    "Accelerated gradient methods for stochastic optimization and online learning."
    Advances in Neural Information Processing Systems. 2009.
    """

    def __init__(
            self,
            model,
            t=0,
            b=1,
            alpha=0.9,
            online=False
    ):
        Optimizer.__init__(
            self,
            model,
            t=t,
        )
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
            self.state['s'] = 1
        else:
            self.hyper_parameter['mu'] = 0
        if isinstance(model, Differentiable):
            self.hyper_parameter['L'] = model.gamma
        else:
            raise ValueError("The loss function must be differentiable")
        self.hyper_parameter['b'] = b
        self.state['alpha'] = alpha
        self.state['step_size'] = None
        self.state['mid_param'] = {}
        self.state['momentum'] = {}
        self.hyper_parameter['online'] = online

    def pre_fb_op(self, x, t):
        if self.hyper_parameter['online']:
            if self.hyper_parameter['mu'] > 0:
                self.state['step_size'] = self.state['alpha'] + self.hyper_parameter['mu'] * self.t \
                                          + self.hyper_parameter['L'] + max(
                    self.hyper_parameter['mu'] - self.hyper_parameter['L'], 0
                ) / self.state['alpha']
            else:
                self.state['step_size'] \
                    = self.hyper_parameter['L'] * (self.state['alpha'] * np.sqrt(self.t - 1) + 1)
        else:
            if self.hyper_parameter['mu'] > 0:
                s = self.state['s']
                self.state['alpha'] = np.sqrt(s + s ** 2 / 4) - s / 2
                self.state['step_size'] = self.hyper_parameter['L'] + self.hyper_parameter['mu'] / s
                self.state['s'] *= (1 - self.state['alpha'])
            else:
                self.state['alpha'] = 2 / (self.t + 2)
                self.state['step_size'] = self.hyper_parameter['b'] * np.power(self.t + 1, 1.5) + self.hyper_parameter[
                    'L']
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, State):
                for key in layer.update_set:
                    self.state['mid_param'][str(i) + key] \
                        = (1 - self.state['alpha']) * layer.param[key].param \
                          + self.state['alpha'] * self.state['momentum'].get(
                        str(i) + key,
                        np.zeros_like(layer.param[key].param)
                    )
                    layer.param[key].param = self.state['mid_param'][str(i) + key].copy()

    def rule(self, i, key, layer):

        grad = layer.param[key].grad

        layer.param[key].param -= grad / self.state['step_size']
        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, 1 / self.state['step_size']
            )
        self.state['momentum'][str(i) + key] \
            = self.state['momentum'].get(str(i) + key, np.zeros_like(grad)) \
              - (
                    self.state['step_size'] * (self.state['mid_param'][str(i) + key] - layer.param[key].param)
                    + self.hyper_parameter['mu'] * (
                        self.state['momentum'].get(str(i) + key, np.zeros_like(grad)) -
                        self.state['mid_param'][str(i) + key])
                ) / (self.state['step_size'] * self.state['alpha'] + self.hyper_parameter['mu'])
