from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.models.components import State


class BaseModel:
    def __init__(self, layers, last_layer):
        self.layers = layers
        self.last_layer = last_layer

    def predict(self, x, *args, **kwargs):
        x = np.asarray(x)
        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
        return self.last_layer.predict(x, *args, **kwargs)

    def loss(self, x, t, *args, **kwargs):
        reg = 0
        x, t = np.asarray(x), np.asarray(t)
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

    def evaluate_model(self, test_iter, show=False):
        raise NotImplementedError()


class Classifier(BaseModel):
    def evaluate_model(self, test_iter, show=True):
        accuracy = 0
        sample_num = 0
        for page in test_iter.pages:
            x, t = zip(*list(page))
            x = np.asarray(x)
            t = np.asarray(t)
            t = t.reshape(t.size)
            y = self.predict(x, train_flg=False)
            if y.size != t.size:
                y = np.argmax(y, axis=1).reshape(t.size)
                sample_num += x.shape[0]
            else:
                tmp = np.zeros_like(y)
                tmp[y > 0.5] = 1
                tmp[y <= 0.5] = 0
                y = tmp.reshape(tmp.size)
                sample_num += y.size
            accuracy += np.sum(y == t)
        if show:
            print('=== Accuracy: {}'.format(accuracy / sample_num))
        return accuracy / sample_num


class Regression(BaseModel):
    def evaluate_model(self, test_iter, show=False):
        error = 0
        sample_num = 0
        for page in test_iter.pages:
            x, t = zip(*list(page))
            x = np.asarray(x)
            t = np.asarray(t)
            y = self.predict(x, train_flg=False)
            error += np.linalg.norm(t - y) ** 2
            sample_num += x.shape[0]
        if show:
            print('=== RMSE: {}'.format(np.sqrt(error / sample_num)))
        return np.sqrt(error / sample_num)
