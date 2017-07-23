from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.functions import Differentiable
from oml.optimizers.optimizer import Optimizer
from oml.models.components import State
from oml.models.components import ProximalOracle


class Svrg(Optimizer):
    """
    Johnson, Rie, and Tong Zhang.
    "Accelerating stochastic gradient descent using predictive variance reduction."
    Advances in neural information processing systems. 2013.
    Reddi, Sashank J., et al.
    "Stochastic variance reduction for nonconvex optimization."
    International conference on machine learning. 2016.
    Reddi, Sashank J., et al.
    "Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization."
     Advances in Neural Information Processing Systems (2016): 1145-1153.
    """

    def __init__(self, model, t=0, step_size=0.01):
        Optimizer.__init__(self, model, t, num_of_target=1)
        if isinstance(model, Differentiable):
            self.hyper_parameter['step_size'] = step_size / (3 * model.gamma)
        else:
            self.hyper_parameter['step_size'] = step_size
        self.state['total_grad'] = {}
        self.state['last_epoch_param'] = {}
        self.state['last_epoch_grad'] = {}
        self.state['last_iter_param'] = {}

    def optimize(self, train_iter, test_iter, epoch=20, max_iter=None, show_loss=False, show_evaluation=False):

        init_t = self.t

        for current_epoch in range(epoch):

            # compute total grad
            iter_num = 0
            loss = None

            for page in train_iter.pages:
                x, t = self.page2data(page)
                iter_num += 1
                self.model.loss(x, t)
                self.model.compute_grad()
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, State):
                    for key in layer.param.keys():
                        self.state['total_grad'][str(i) + key] = layer.param[key].grad / iter_num
                        self.state['last_epoch_param'][str(i) + key] = layer.param[key].param
            self.model.clear_grad()
            train_iter.initialize()

            # update step

            for page in train_iter.pages:
                x, t = self.page2data(page)
                self.t += 1

                # cal last epoch grad
                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, State):
                        for key in layer.param.keys():
                            self.state['last_iter_param'][str(i) + key] = layer.param[key].param
                            layer.param[key].param = self.state['last_epoch_param'][str(i) + key]
                self.model.loss(x, t)
                self.model.compute_grad()
                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, State):
                        for key in layer.param.keys():
                            self.state['last_epoch_grad'][key] = layer.param[key].grad
                            layer.param[key].param = self.state['last_iter_param'][str(i) + key]
                self.model.clear_grad()

                # parameter update
                loss = self.model.loss(x, t)

                if show_loss:
                    print('=== loss: {}'.format(loss))

                self.model.compute_grad()

                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, State):
                        for key in layer.param.keys():
                            v = layer.param[key].grad - self.state['last_epoch_grad'][key] \
                                + self.state['total_grad'][str(i) + key]
                            layer.param[key].param -= self.hyper_parameter['step_size'] * v
                            if isinstance(layer.param[key], ProximalOracle):
                                layer.param[key].param = layer.param[key].reg.proximal(
                                    layer.param[key].param,
                                    self.hyper_parameter['step_size']
                                )
                            self.state['last_iter_param'][key] = layer.param[key].param

                if max_iter and max_iter < self.t - init_t:
                    break

                self.model.clear_grad()
            if show_evaluation:
                self.evaluation.append(self.model.evaluate_model(test_iter))
                self.loss.append(loss)
            train_iter.initialize()
            test_iter.initialize()

        print('=== Final Evaluation ===')
        self.model.evaluate_model(test_iter)
        test_iter.initialize()
