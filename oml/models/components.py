from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

import numpy as np

from oml import functions as F
from .regularizers import Nothing


class Layer:
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, dout):
        raise NotImplementedError()


class LastLayer:
    def forward(self, x, t, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
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
    def __init__(self, shape=None, reg=Nothing(), param=None):
        ProximalOracle.__init__(self, reg)
        if shape is not None:
            FirstOrderOracle.__init__(self, np.zeros(shape))
            self.param = np.random.normal(size=shape) * 1e-7
        elif param is not None:
            FirstOrderOracle.__init__(self, np.zeros_like(param))
            self.param = param
        else:
            raise ValueError("Fail to initialize")


class State:
    def __init__(self, param: dict):
        self.param = param
        self.update_set = set()


class Affine(Layer, State):
    def __init__(self, input_size, output_size, reg=Nothing()):
        State.__init__(self, {
            'w': Param((input_size, output_size), reg=reg),
            'b': Param((1, output_size))
        })
        self.update_set.add('w')
        self.update_set.add('b')
        self.x = None
        self.original_x_shape = None

    def forward(self, x, *args, **kwargs):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.param['w'].param) + self.param['b'].param

    def backward(self, dout):
        dx = np.dot(dout, self.param['w'].param.T)
        self.param['w'].grad += np.dot(self.x.T, dout)
        self.param['b'].grad += np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


class FactorizationMachine(Layer, State):
    """
    Rendle, Steffen.
    "Factorization machines."
    Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.

    accelerate by Dynamic Programming
    Blondel, Mathieu, et al.
    "Higher-order factorization machines."
    Advances in Neural Information Processing Systems (2016): 3351-3359.
    """

    def __init__(self, rank_list, reg=Nothing()):
        State.__init__(self, {'b': Param(shape=(1,))})
        self.variance_reg = reg
        self.rank_list = rank_list
        self.x = None
        self.order = len(rank_list)
        self.update_set.add('b')
        self.dp_table = {}

    def forward(self, x: List[dict], *args, **kwargs):
        self.x = x

        res = np.ones((len(x), 1)) * self.param['b'].param

        for n in range(len(x)):

            key_list = list(x[n].keys())

            # dp
            for o in range(self.order):

                a = np.zeros((o + 2, len(key_list) + 1, self.rank_list[o]))
                a[0, :, :] = 1

                for t in range(1, o + 2):
                    for j in range(t, len(key_list) + 1):
                        key = key_list[j - 1]
                        if key not in self.param:
                            self.update_set.add('{}-{}'.format(o, key))
                            self.param['{}-{}'.format(o, key)] = Param(
                                shape=(self.rank_list[o],),
                                reg=self.variance_reg
                            )
                        a[t, j, :] = a[t, j - 1, :] \
                                     + self.param['{}-{}'.format(o, key)].param * x[n][key] * a[t - 1, j - 1, :]

                res[n] += np.sum(a[o + 1, len(key_list), :])

                self.dp_table[(n, o)] = a

        return res

    def backward(self, dout: np.ndarray):

        self.param['b'].grad += np.sum(dout, axis=0) * 1

        for n in range(len(self.x)):

            key_list = list(self.x[n].keys())

            for o in range(self.order):

                a = self.dp_table[(n, o)]

                a_ = np.zeros((o + 2, len(key_list) + 1, self.rank_list[o]))
                a_[o + 1, len(key_list), self.rank_list[o] - 1] = 1

                for t in range(o, 0, -1):
                    for j in range(len(key_list) - 1, t - 1, -1):
                        key = key_list[j - 1]
                        a_[t, j, :] = a_[t, j + 1, :] + a_[t + 1, j + 1, :] \
                                                        * self.param['{}-{}'.format(o, key)].param * self.x[n][key]

                for j in range(len(key_list)):
                    for t in range(1, o + 2):
                        self.param['{}-{}'.format(o, key_list[j])].grad += a_[t, j + 1] * a[t - 1, j] * dout[n]

        self.dp_table.clear()
        return None


class Sigmoid(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x, *args, **kwargs):
        self.x = x
        return F.sigmoid(self.x)

    def backward(self, dout):
        return np.multiply(np.multiply(dout, 1.0 - F.sigmoid(self.x)), F.sigmoid(self.x))


class Relu(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x, *args, **kwargs):
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

    def forward(self, x, *args, **kwargs):
        self.x = x
        return F.softplus(self.x)

    def backward(self, dout):
        return np.multiply(F.sigmoid(self.x), dout)


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

    def forward(self, x, t, *args, **kwargs):
        self.y = self.predict(x, *args, **kwargs)
        self.t = t.reshape(self.y.shape)
        return F.mean_squared(self.y, self.t)

    def predict(self, x, *args, **kwargs):
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

    def predict(self, x, *args, **kwargs):
        self.x = x
        return np.exp(self.x)

    def forward(self, x, t, *args, **kwargs):
        self.y = self.predict(x, *args, **kwargs)
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

    def predict(self, x, *args, **kwargs):
        self.x = x
        return F.softmax(x)

    def forward(self, x, t, *args, **kwargs):
        self.t = t
        self.y = self.predict(x, *args, **kwargs)
        if self.t.size == self.y.size:
            self.t = self.t.reshape(self.t.size)
            self.y = self.y.reshape(self.y.size)
        return F.cross_entropy(self.y, self.t)

    def backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
            dx = dx.reshape(batch_size, -1)
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t.astype(int)] -= 1
            dx = dx / batch_size
        return dx


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(n * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1
    col = col.reshape(n, out_h, out_w, c, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:h + pad, pad:w + pad]


class Convolution(Layer, State):
    def __init__(self, weight_shape, bias_shape, stride=1, pad=0, reg=Nothing()):
        Layer.__init__(self)
        State.__init__(self, {
            'w': Param(weight_shape, reg=reg),
            'b': Param(bias_shape)
        })
        self.update_set.add('w')
        self.update_set.add('b')
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col_x = None
        self.col_w = None

    def forward(self, x, *args, **kwargs):
        fn, c, fh, fw = self.param['w'].param.shape
        n, c, h, w = x.shape
        out_h = 1 + int((h + 2 * self.pad - fh) / self.stride)
        out_w = 1 + int((w + 2 * self.pad - fw) / self.stride)

        col = im2col(x, fh, fw, self.stride, self.pad)
        col_w = self.param['w'].param.reshape(fn, -1).T

        out = np.dot(col, col_w) + self.param['b'].param
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col_x = col
        self.col_w = col_w

        return out

    def backward(self, dout):
        fn, c, fh, fw = self.param['w'].param.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, fn)

        self.param['b'].grad += np.sum(dout, axis=0)
        grad = np.dot(self.col_x.T, dout)
        self.param['w'].grad += grad.transpose((1, 0)).reshape(fn, c, fh, fw)

        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)

        return dx


class Pooling(Layer):
    def __init__(self, pool_size, stride=1, pad=0):
        Layer.__init__(self)
        self.pool_h = pool_size[0]
        self.pool_w = pool_size[1]
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x, *args, **kwargs):
        n, c, h, w = x.shape
        out_h = int(1 + (h - self.pool_h) / self.stride)
        out_w = int(1 + (w - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)

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


class Dropout(Layer):
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        Layer.__init__(self)
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, *args, **kwargs):
        if kwargs.get('train_flg', True):
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization(Layer, State):
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma_initializer=np.ones, beta_initializer=np.zeros, momentum=0.9, running_mean=None,
                 running_var=None):
        Layer.__init__(self)
        State.__init__(self, {
            'gamma': None,
            'beta': None,
        })
        if gamma_initializer is None:
            gamma_initializer = np.ones_like
        if beta_initializer is None:
            beta_initializer = np.zeros_like
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None

    def forward(self, x, *args, **kwargs):
        if self.param['gamma'] is None:
            self.param['gamma'] = Param(param=self.gamma_initializer((x.size // x.shape[0],), dtype=np.float32))
        if self.param['beta'] is None:
            self.param['beta'] = Param(param=self.beta_initializer((x.size // x.shape[0],), dtype=np.float32))

        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, kwargs.get('train_flg', True))
        if kwargs.get('train_flg', True):
            self.update_set.add('gamma')
            self.update_set.add('beta')
        else:
            self.update_set.clear()

        return out.reshape(*self.input_shape)

    def __forward(self, x, *args, **kwargs):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if kwargs.get('train_flg', True):
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.param['gamma'].param * xn + self.param['beta'].param
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.param['gamma'].param * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.param['gamma'].grad = dgamma
        self.param['beta'].grad = dbeta

        return dx
