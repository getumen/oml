from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division


from oml.models.components import State


class Optimizer:
    def __init__(
            self,
            model,
            t=0,
            num_of_target=1
    ):
        self.t = t

        self.model = model
        self.state = {}
        self.hyper_parameter = {}
        self.num_of_target = num_of_target
        self.loss = []
        self.evaluation = []

    def optimize(self, train_iter, test_iter, epoch=20, max_iter=None, show_loss=False, show_evaluation=False):

        init_t = self.t

        for current_epoch in range(epoch):

            self.evaluation.append(self.model.evaluate_model(test_iter, show=show_evaluation))
            train_iter.initialize()
            test_iter.initialize()

            self.pre_epoch(train_iter, test_iter)

            for page in train_iter.pages:
                x, t = zip(*list(page))
                self.t += 1

                self.pre_fb_op(x, t)

                loss = self.model.loss(x, t)
                self.loss.append(loss)

                if show_loss:
                    print('=== loss: {}'.format(loss))

                self.model.compute_grad()

                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, State):
                        for key in layer.update_set:
                            self.rule(i, key, layer)

                if max_iter and max_iter < self.t - init_t:
                    break

                self.model.clear_grad()

        print('=== Final Evaluation ===')
        self.model.evaluate_model(test_iter, show=True)
        test_iter.initialize()

    def rule(self, i, key, layer):
        raise NotImplementedError()

    def pre_fb_op(self, x, t):
        pass

    def pre_epoch(self, train_iter, test_iter):
        pass
