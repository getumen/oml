from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.models.components import ProximalOracle
from oml.optimizers import optimizer


class Smidas(optimizer.Optimizer):
    """
    Shalev-Shwartz, Shai, and Ambuj Tewari.
    "Stochastic methods for l1-regularized loss minimization."
    Journal of Machine Learning Research 12.Jun (2011): 1865-1892.

    Change how to decide the step size
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
        self.state['theta'] = {}

    def rule(self, i, key, layer):
        grad = layer.param[key].grad
        self.state['theta'][str(i) + key] = \
            self.state['theta'].get(str(i) + key, np.zeros_like(grad)) \
            - self.hyper_parameter['step_size'] * grad / self.t

        if isinstance(layer.param[key], ProximalOracle):
            self.state['theta'][str(i) + key] = layer.param[key].reg.proximal(
                self.state['theta'][str(i) + key], self.hyper_parameter['step_size'] / self.t
            )

        p = 1 + 1 / np.log(grad.size)

        layer.param[key].param = np.sign(self.state['theta'][str(i) + key]) * np.power(
            np.absolute(self.state['theta'][str(i) + key]),
            p - 1
        ) / norm(
            self.state['theta'][str(i) + key],
            p
        ) ** (2 - p)


def norm(x, p):
    return np.sum(np.absolute(x) ** p) ** (1. / p)
