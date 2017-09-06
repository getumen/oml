from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from . import optimizer
from oml.models.components import ProximalOracle

from oml.functions import StrongConvexity, Differentiable
from oml.models.components import State

import warnings


class Rda(optimizer.Optimizer):
    """
    Xiao, Lin.
    "Dual averaging methods for regularized stochastic learning and online optimization."
    Journal of Machine Learning Research 11.Oct (2010): 2543-2596.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['step_size'] = step_size
        self.state['averaged_cumulative_grad'] = {}
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
        else:
            warnings.warn("RDA is not appropriate for optimizing non convex objective")
            self.hyper_parameter['mu'] = None

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        self.state['averaged_cumulative_grad'][str(i) + key] = \
            (
                self.state['averaged_cumulative_grad'].get(str(i) + key, np.zeros_like(grad)) * (self.t - 1)
                + grad
            ) / self.t

        layer.param[key].param = \
            -np.sqrt(self.t) * self.hyper_parameter['step_size'] * self.state['averaged_cumulative_grad'][
                str(i) + key]
        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param, self.hyper_parameter['step_size'] * np.sqrt(self.t)
            )


class AcceleratedRDA(optimizer.Optimizer):
    """
        Xiao, Lin.
        "Dual averaging methods for regularized stochastic learning and online optimization."
        Journal of Machine Learning Research 11.Oct (2010): 2543-2596.
        """

    def __init__(
            self,
            model,
            t=0,
            gamma=2,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.hyper_parameter['gamma'] = gamma
        self.state['A'] = 0
        self.state['v'] = {}
        self.state['last_param'] = {}
        self.state['averaged_cumulative_grad'] = {}

        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
        else:
            warnings.warn("RDA is not appropriate for optimizing non convex objective")
            self.hyper_parameter['mu'] = None
        if isinstance(model, Differentiable):
            self.hyper_parameter['L'] = model.gamma
        else:
            raise ValueError("The loss of model is not differentiable")

    def pre_fb_op(self, train_iter, test_iter):
        self.state['A'] = self.t * (self.t + 1) / 4
        theta = 2 / (self.t + 1)

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, State):
                for key in layer.param.keys():
                    self.state['last_param'][str(i) + key] = layer.param[key].param
                    layer.param[key].param \
                        = (1 - theta) * layer.param[key].param
                    + theta * self.state['v'].get(str(i) + key, np.random.randn(*layer.param[key].param.shape) * 1e-8)

    def rule(self, i, key, layer):
        theta = 2 / (self.t + 1)
        grad = layer.param[key].grad
        beta = self.hyper_parameter['gamma'] * (self.t + 1) * np.sqrt(self.t + 1) / 2
        step_size = self.state['A'] / (self.hyper_parameter['L'] + beta)
        self.state['averaged_cumulative_grad'][str(i) + key] = \
            self.state['averaged_cumulative_grad'].get(str(i) + key, np.zeros_like(grad)) * (1 - theta) + grad * theta

        self.state['v'][str(i) + key] = - step_size * self.state['averaged_cumulative_grad'][str(i) + key]

        if isinstance(layer.param[key], ProximalOracle):
            self.state['v'][str(i) + key] = layer.param[key].reg.proximal(self.state['v'][str(i) + key], step_size)

        layer.param[key].param = \
            (1 - theta) * self.state['last_param'][str(i) + key] + theta * self.state['v'][str(i) + key]
