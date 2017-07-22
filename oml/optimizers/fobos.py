from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.functions import StrongConvexity
from oml.models.components import State
from oml.models.glm import BaseGLM
from oml.optimizers import optimizer
from oml.models.components import ProximalOracle


class Fobos(optimizer.Optimizer):
    def __init__(
            self,
            model: BaseGLM,
            step_size=0.01,
            t=0,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            t=t,
        )
        self.step_size = step_size
        if isinstance(model, StrongConvexity):
            self.mu = model.mu
        else:
            self.mu = 0

    def optimize(self, train_data, test_data, epoch=20, max_iter=None, verbose=False):

        init_t = self.t

        for current_epoch in range(epoch):
            for x, t in train_data:
                self.t += 1
                loss = self.model.loss(x, t)

                if verbose:
                    print('=== loss: {}'.format(loss))

                self.model.compute_grad()

                for layer in self.model.layers:
                    if isinstance(layer, State):
                        for key in layer.param.keys():
                            if self.mu == 0:
                                layer.param[key].param -= self.step_size / np.sqrt(self.t) * layer.param[key].grad
                            else:
                                layer.param[key].param -= layer.param[key].grad / self.t / self.mu

                            if isinstance(layer.param[key], ProximalOracle):
                                layer.param[key].param = layer.param[key].reg.proximal(layer.param[key].param,
                                                                                       self.step_size / np.sqrt(self.t))

                if max_iter and max_iter < self.t - init_t:
                    break

                self.model.clear_grad()

            self.model.evaluate_model(test_data)
            train_data.initialize()
            test_data.initialize()
