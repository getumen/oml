from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np
from oml.models.components import State


class BaseModel:
    def __init__(self, layers, last_layer):
        self.layers = layers
        self.last_layer = last_layer

    def predict(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
        return self.last_layer.predict(x, *args, **kwargs)

    def loss(self, x, t, *args, **kwargs):
        reg = 0
        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
            if isinstance(layer, State):
                for key in layer.param.keys():
                    reg += layer.param[key].reg.apply(layer.param[key].param)
        return self.last_layer.forward(x, t, *args, **kwargs) + reg

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


class Classifier(BaseModel):
    def evaluate_model(self, test_iter):
        accuracy = 0
        sample_num = 0
        for page in test_iter.pages:
            data = np.matrix(list(page))
            x, t = data[:, :-1], data[:, -1]
            t = np.asarray(t).reshape(t.size)
            y = self.predict(x, train_flg=False)
            y = np.argmax(y, axis=1)
            accuracy += np.sum(y == t)
            sample_num += x.shape[0]

        print('=== Accuracy: {}'.format(accuracy / sample_num))


class Regression(BaseModel):
    def evaluate_model(self, test_iter):
        error = 0
        sample_num = 0
        for page in test_iter.pages:
            data = np.matrix(list(page))
            x, t = data[:, :-1], data[:, -1]
            error += np.linalg.norm(t - self.predict(x, train_flg=False)) ** 2
            sample_num += x.shape[0]

        print('=== RMSE: {}'.format(np.sqrt(error / sample_num)))
