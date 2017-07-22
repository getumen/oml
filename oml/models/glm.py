from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.models.components import Affine, Gauss, Softmax, State
from oml.models.regulizers import L1, L2Sq
import numpy as np

"""
Generalized Linear Models
"""


class BaseGLM:
    def __init__(self, layers, last_layer):
        self.layers = layers
        self.last_layer = last_layer

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.predict(x)

    def loss(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.forward(x, t)

    def clear_grad(self):
        for layer in self.layers:
            if isinstance(layer, State):
                for value in layer.param.values():
                    value.clear_grad()

    def compute_grad(self):
        dout = self.last_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def evaluate_model(self, test_iter):
        raise NotImplementedError()


class LinearRegression(BaseGLM):
    def __init__(
            self,
            input_size,
            output_size,
            reg=L2Sq(param=0.01)
    ):
        BaseGLM.__init__(
            self,
            [
                Affine(input_size, output_size, reg=reg)
            ],
            Gauss(),
        )

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.predict(x)

    def loss(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)
        return self.last_layer.forward(x, t)

    def evaluate_model(self, test_iter):
        error = 0
        sample_num = 0
        for x, t in test_iter:
            error += np.linalg.norm(t - self.predict(x)) ** 2
            sample_num += 1

        print('=== RMSE: {}'.format(np.sqrt(error / sample_num)))


class SoftmaxRegression(BaseGLM):
    def __init__(
            self,
            input_size,
            output_size,
            reg=L1(param=0.01)
    ):
        BaseGLM.__init__(
            self,
            [
                Affine(input_size, output_size, reg=reg)
            ],
            Softmax(),
        )

    def evaluate_model(self, test_iter):
        accuracy = 0
        sample_num = 0
        for x, t in test_iter:
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            if t.ndim >= 2:
                t = np.argmax(t, axis=1)
            accuracy += np.sum(y == t) / float(x.shape[0])
            sample_num += 1

        print('=== Accuracy: {}'.format(accuracy/sample_num))
