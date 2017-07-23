from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from oml.models.components import State


class Optimizer:
    def __init__(
            self,
            model,
            t=0,
            num_of_t=1
    ):
        self.t = t

        self.model = model
        self.state = {}
        self.hyper_parameter = {}
        self.num_of_t = num_of_t

    def page2data(self, page):
        data = np.matrix(list(page))
        if self.num_of_t == 1:
            x, t = data[:, :-self.num_of_t], np.squeeze(np.asarray(data[:, -self.num_of_t]))
        else:
            x, t = data[:, :-self.num_of_t], data[:, -self.num_of_t]
        return x, t

    def optimize(self, train_iter, test_iter, epoch=20, max_iter=None, show_loss=False, show_evaluation=False):

        init_t = self.t

        for current_epoch in range(epoch):
            for page in train_iter.pages:
                x, t = self.page2data(page)
                self.t += 1
                loss = self.model.loss(x, t)

                if show_loss:
                    print('=== loss: {}'.format(loss))

                self.model.compute_grad()

                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, State):
                        for key in layer.param.keys():
                            self.rule(i, key, layer)

                if max_iter and max_iter < self.t - init_t:
                    break

                self.model.clear_grad()
            if show_evaluation:
                self.model.evaluate_model(test_iter)
            train_iter.initialize()
            test_iter.initialize()

        print('=== Final Evaluation ===')
        self.model.evaluate_model(test_iter)
        test_iter.initialize()

    def rule(self, i, key, layer):
        raise NotImplementedError()
