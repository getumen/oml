from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.sparse import csr_matrix

from oml import functions as F
from oml.model import FirstOrderOracle


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


class Affine(FirstOrderOracle, Layer):
    def __init__(self, input_size, output_size, sparse=False):
        FirstOrderOracle.__init__(self)
        if sparse:
            self.param['w'] = csr_matrix((input_size, output_size))
            self.param['b'] = csr_matrix(output_size)
        else:
            self.param['w'] = np.zeros((input_size, output_size))
            self.param['b'] = np.zeros(output_size)
        self.x = None
        self.original_x_shape = None
        self.grad['w'] = None
        self.grad['b'] = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.param['w']) + self.param['b']

    def backward(self, dout):
        dx = np.dot(dout, self.param['w'].T)
        self.grad['w'] = np.dot(self.x.T, dout)
        self.grad['b'] = np.sum(dout, axis=0)
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
        return np.multiply(np.exp(self.x)/ np.add(1, np.exp(self.x)), dout)


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
        batch_size = self.t.shape[0]
        return np.multiply(self.y - self.t, 1/batch_size)


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
        batch_size = self.t.shape[0]
        return np.multiply(self.y - self.t, dout/batch_size)


class Softmax(LastLayer):
    """
    softmax with Cross Entropy
    """
    def __init__(self):
        self.y = None
        self.t = None

    def predict(self, x):
        self.x = x
        return F.softmax(x)

    def forward(self, x, t):
        self.t = t
        self.y = self.predict(x)
        return F.cross_entropy(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = np.multiply(self.y - self.t, dout / batch_size)
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = np.multiply(dx, dout / batch_size)

        return dx
