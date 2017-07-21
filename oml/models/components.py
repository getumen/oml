from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np
from scipy.sparse import csr_matrix

from oml import functions as F


class StateMinxin:
    def __init__(self, input_size, output_size, sparse=False):
        if sparse:
            self.w = csr_matrix((input_size, output_size))
            self.b = csr_matrix(output_size)
        else:
            self.w = np.zeros((input_size, output_size))
            self.b = np.zeros(output_size)
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None


class Linear(StateMinxin):
    def __init__(self, input_size, output_size, sparse=False):
        StateMinxin.__init__(self, input_size, output_size, sparse)

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.w) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return F.sigmoid(self.x)

    def backward(self, dout):
        return np.multiply(np.multiply(dout, 1.0 - F.sigmoid(self.x)), F.sigmoid(self.x))


class Relu:
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


class SoftPlus:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return F.softplus(self.x)

    def backward(self, dout):
        return np.multiply(np.exp(self.x)/ np.add(1, np.exp(self.x)), dout)


class Gauss:
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
        self.x = x
        self.y = x
        self.t = t
        return F.mean_squared(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return np.multiply(self.y - self.t, dout/batch_size)


class Poisson:
    """
    Poisson with negative log likelihood
    Poisson Regression
    ignore constant
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.x = x
        self.y = np.exp(x)
        self.t = t
        return np.sum(self.y - np.multiply(t, self.x))

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return np.multiply(self.y - self.t, dout/batch_size)


class Softmax:
    """
    softmax with Cross Entropy
    """
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = F.softmax(x)
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
