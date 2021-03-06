from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.functions import Differentiable, StrongConvexity
from .optimizer import Optimizer
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

        self.hyper_parameter['step_size'] = step_size
        self.state['total_grad'] = {}
        self.state['last_epoch_param'] = {}
        self.state['last_epoch_grad'] = {}
        self.state['last_iter_param'] = {}

    def pre_epoch(self, train_iter, test_iter):
        # compute total grad
        iter_num = 0
        loss = None

        for page in train_iter.pages:
            x, t = zip(*list(page))

            iter_num += 1
            self.model.loss(x, t)
            self.model.compute_grad()

        if isinstance(self.model, Differentiable) and isinstance(self.model, StrongConvexity):
            self.hyper_parameter['step_size'] = 1 / (3 * self.model.gamma * iter_num)

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, State):
                for key in layer.update_set:
                    self.state['total_grad'][str(i) + key] = layer.param[key].grad / iter_num
                    self.state['last_epoch_param'][str(i) + key] = layer.param[key].param
        self.model.clear_grad()
        train_iter.initialize()

        # update step

    def pre_fb_op(self, x, t):
        # cal last epoch grad
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, State):
                for key in layer.update_set:
                    self.state['last_iter_param'][str(i) + key] = layer.param[key].param
                    layer.param[key].param = self.state['last_epoch_param'][str(i) + key]
        self.model.loss(x, t)
        self.model.compute_grad()
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, State):
                for key in layer.param.keys():
                    self.state['last_epoch_grad'][str(i) + key] = layer.param[key].grad
                    layer.param[key].param = self.state['last_iter_param'][str(i) + key]
        self.model.clear_grad()

    def rule(self, i, key, layer):
        v = layer.param[key].grad - self.state['last_epoch_grad'][str(i) + key] \
            + self.state['total_grad'][str(i) + key]
        layer.param[key].param -= self.hyper_parameter['step_size'] * v
        if isinstance(layer.param[key], ProximalOracle):
            layer.param[key].param = layer.param[key].reg.proximal(
                layer.param[key].param,
                self.hyper_parameter['step_size']
            )
        self.state['last_iter_param'][str(i) + key] = layer.param[key].param
