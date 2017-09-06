from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.functions import StrongConvexity
from oml.models.components import ProximalOracle
from . import optimizer


class Fobos(optimizer.Optimizer):
    """
    Duchi, John, and Yoram Singer.
    "Efficient online and batch learning using forward backward splitting."
    Journal of Machine Learning Research 10.Dec (2009): 2899-2934.

    If objective is non-convex, this becomes SGD with fixed step size.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            num_of_target=1
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
            num_of_target=num_of_target,
        )
        self.hyper_parameter['step_size'] = step_size
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
        else:
            self.hyper_parameter['mu'] = None

    def rule(self, i, key, layer):
        if self.hyper_parameter['mu'] is None:
            step_size = self.hyper_parameter['step_size']
        elif self.hyper_parameter['mu'] == 0:
            step_size = self.hyper_parameter['step_size'] / np.sqrt(self.t)
        else:
            step_size = 1 / self.t / self.hyper_parameter['mu']
        layer.param[key].param -= step_size * layer.param[key].grad

        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param,
                step_size,
            )


class SGDWithNoise(optimizer.Optimizer):
    """
    TODO: read this:
    Zhang, Yuchen, Percy Liang, and Moses Charikar.
    "A Hitting Time Analysis of Stochastic Gradient Langevin Dynamics."
    COLT (2017) best paper.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            xi=1,
            num_of_target=1
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
            num_of_target=num_of_target,
        )
        self.hyper_parameter['step_size'] = step_size
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
        else:
            self.hyper_parameter['mu'] = None
        self.hyper_parameter['xi'] = xi

    def rule(self, i, key, layer):
        grad = layer.param[key].grad + np.random.normal(
            0,
            np.sqrt(2 * self.hyper_parameter['step_size'] / self.hyper_parameter['xi']),
            size=layer.param[key].grad.shape
        )
        if isinstance(layer.param[key], ProximalOracle):
            grad += layer.param[key].reg.sub_differential(layer.param[key].param)
        layer.param[key].param -= self.hyper_parameter['step_size'] * grad

