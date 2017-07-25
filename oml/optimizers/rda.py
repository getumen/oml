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

