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
    def __init__(self, shape, reg=Nothing(), sparse=False):
        ProximalOracle.__init__(self, reg)
        FirstOrderOracle.__init__(self, np.zeros(shape))
        if sparse:
            self.param = csr_matrix(shape)
        else:
            self.param = np.zeros(shape)


class State:
    def __init__(self, param: dict):
        self.param = param


class Affine(Layer, State):
    def __init__(self, input_size, output_size, reg=Nothing(), sparse=False):
        State.__init__(self, {
            'w': Param((input_size, output_size), sparse=sparse, reg=reg),
            'b': Param((1, output_size), sparse=sparse)
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
        self.t = t.reshape(self.y.shape)
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
        self.t = t.reshape(self.y.shape)
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


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Convolution(Layer, State):
    def __init__(self, weight_shape, bias_shape, stride=1, pad=0, sparse=True, reg=Nothing()):
        Layer.__init__(self)
        State.__init__(self, {
            'w': Param(weight_shape, sparse=sparse, reg=reg),
            'b': Param(bias_shape, sparse=sparse)
        })
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.param['w'].param.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.param['w'].param.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.param['b'].param
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.param['w'].param.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.param['b'].grad += np.sum(dout, axis=0)
        self.param['w'].grad += np.dot(self.col.T, dout)
        self.param['w'].grad = self.param['w'].grad.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
