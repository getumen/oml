from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.sparse import csr_matrix

from oml import functions as F
from oml.models.regulizers import Reg, Nothing


class Layer:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dout):
        raise NotImplementedError()


class LastLayer:
    def forward(self, x, t):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()


class FirstOrderOracle:
    def __init__(self, grad: np.ndarray):
        self.grad = grad

    def clear_grad(self):
        self.grad = np.zeros_like(self.grad)


class ProximalOracle:
    def __init__(self, reg):
        self.reg = reg


class Param(FirstOrderOracle, ProximalOracle):
    def __init__(self, input_size, output_size, reg=Nothing(), sparse=False):
        ProximalOracle.__init__(self, reg)
        FirstOrderOracle.__init__(self, np.zeros((input_size, output_size)))
        if sparse:
            self.param = csr_matrix((input_size, output_size))
        else:
            self.param = np.zeros((input_size, output_size))


class State:
    def __init__(self, param: dict):
        self.param = param


class Affine(Layer, State):
    def __init__(self, input_size, output_size, reg=Nothing(), sparse=False):
        State.__init__(self, {
            'w': Param(input_size, output_size, sparse=sparse, reg=reg),
            'b': Param(1, output_size, sparse=sparse)
        })

        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.param['w'].param) + self.param['b'].param

    def backward(self, dout):
        dx = np.dot(dout, self.param['w'].param.T)
        self.param['w'].grad += np.dot(self.x.T, dout)
        self.param['b'].grad += np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


class Sigmoid(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return F.sigmoid(self.x)

    def backward(self, dout):
        return np.multiply(np.multiply(dout, 1.0 - F.sigmoid(self.x)), F.sigmoid(self.x))


class Relu(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SoftPlus(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return F.softplus(self.x)

    def backward(self, dout):
        return np.multiply(np.exp(self.x) / np.add(1, np.exp(self.x)), dout)


class Gauss(LastLayer):
    """
    Gauss with negative log likelihood
    Linear Regression
    ignore constant
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = self.predict(x)
        self.t = t
        return F.mean_squared(self.y, self.t)

    def predict(self, x):
        self.x = x
        return self.x

    def backward(self):
        batch_size = self.x.shape[0]
        return np.multiply(self.y - self.t, 1 / batch_size)


class Poisson(LastLayer):
    """
    Poisson with negative log likelihood
    Poisson Regression
    ignore constant
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.t = None

    def predict(self, x):
        self.x = x
        return np.exp(self.x)

    def forward(self, x, t):
        self.y = self.predict(x)
        self.t = t
        return np.sum(self.y - np.multiply(t, self.x))

    def backward(self):
        batch_size = self.x.shape[0]
        return np.multiply(self.y - self.t, 1 / batch_size)


class Softmax(LastLayer):
    """
    softmax with Cross Entropy
    """

    def __init__(self):
        self.y = None
        self.t = None
        self.x = None

    def predict(self, x):
        self.x = x
        return F.softmax(x)

    def forward(self, x, t):
        self.t = t
        self.y = self.predict(x)
        return F.cross_entropy(self.y, self.t)

    def backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t.astype(int)] -= 1
            dx = dx / batch_size
        return dx
