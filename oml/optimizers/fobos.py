from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from oml import optimizer
from oml.models.glm import BaseGLM
from oml.functions import StrongConvexity


class Fobos(optimizer.Optimizer):
    def __init__(
            self,
            model: BaseGLM,
            train_data,
            test_data,
            max_iter=None,
            step_size=0.01,
            t=0,
            verbose=True,
    ):
        optimizer.Optimizer.__init__(
            self,
            model,
            train_data,
            test_data,
            t=t,
            verbose=verbose
        )
        self.max_iter = max_iter
        self.step_size = step_size
        if isinstance(model, StrongConvexity):
            self.mu = model.mu
        else:
            self.mu = 0

    def optimize(self):
        self.t += 1
        for x, t in self.train_data:
            loss = self.model.loss(x, t)

            if self.verbose:
                print('=== loss: {}'.format(loss))

            for oracle in self.model.get_oracle():
                for key in oracle.param.keys():
                    if self.mu == 0:
                        oracle.param[key] -= self.step_size * oracle.grad[key] / np.sqrt(self.t)
                    else:
                        oracle.param[key] -= oracle.grad[key] / self.t / self.mu
            if self.max_iter and self.max_iter < self.t:
                break
