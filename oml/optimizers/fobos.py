from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.functions import StrongConvexity
from oml.optimizers import optimizer
from oml.models.components import ProximalOracle


class Fobos(optimizer.Optimizer):
    """
    Duchi, John, and Yoram Singer.
    "Efficient online and batch learning using forward backward splitting."
    Journal of Machine Learning Research 10.Dec (2009): 2899-2934.
    """

    def __init__(
            self,
            model,
            step_size=0.01,
            t=0,
            num_of_t=1
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
            num_of_t=num_of_t,
        )
        self.hyper_parameter['step_size'] = step_size
        if isinstance(model, StrongConvexity):
            self.hyper_parameter['mu'] = model.mu
        else:
            self.hyper_parameter['mu'] = 0

    def rule(self, key, layer):
        if self.hyper_parameter['mu'] == 0:
            layer.param[key].param -= self.hyper_parameter['step_size'] / np.sqrt(self.t) * layer.param[key].grad
        else:
            layer.param[key].param -= layer.param[key].grad / self.t / self.hyper_parameter['mu']
        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param,
                self.hyper_parameter['step_size'] / np.sqrt(self.t)
            )