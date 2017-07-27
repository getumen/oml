from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.sparse import csr_matrix

from oml import functions as F
from oml.models.regulizers import Nothing


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
    def __init__(self, shape=None, reg=Nothing(), sparse=False, param=None):
        ProximalOracle.__init__(self, reg)
        if shape is not None:
            FirstOrderOracle.__init__(self, np.zeros(shape))
            if sparse:
                self.param = csr_matrix(np.zeros(shape))
            else:
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
    def __init__(self, input_size, output_size, reg=Nothing(), sparse=False):
        State.__init__(self, {
            'w': Param((input_size, output_size), sparse=sparse, reg=reg),
            'b': Param((1, output_size), sparse=sparse)
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
    def __init__(self, rank=5, input_size=10, output_size=1, input_bias_reg=Nothing(), variance_reg=Nothing()):
        State.__init__(self, {
            'b': Param(shape=(output_size,))
        })
        self.input_bias_reg = input_bias_reg
        self.variance_reg = variance_reg
        self.rank = rank
        self.input_size = input_size
        self.output_size = output_size
        self.x = None
        self.x_original_shape = None

    def forward(self, x: np.ndarray, *args, **kwargs):
        x = x.astype(int)
        res = np.zeros((x.shape[0], self.output_size))
        self.x = x
        self.x_original_shape = x.shape

        if x.ndim == 3:
            self.update_set.add('b')
            for n in range(x.shape[0]):
                res[n] += self.param['b'].param
                vx = np.zeros((self.output_size, self.rank))
                for i in range(x.shape[1]):

                    channel = x[n, i, 0]
                    index = x[n, i, 1]
                    value = x[n, i, 2]
                    w_key = 'w-{}-{}'.format(channel, index)
                    v_key = 'v-{}-{}'.format(channel, index)

                    if w_key not in self.param:
                        self.param[w_key] = \
                            Param(param=np.random.randn(self.output_size) * 1e-8, reg=self.input_bias_reg)

                    w_i = self.param[w_key].param

                    if v_key not in self.param:
                        self.param[v_key] = \
                            Param(param=np.random.randn(self.output_size, self.rank) * 1e-8, reg=self.variance_reg)
                    v_i = self.param[v_key].param
                    vx += v_i * value
                    res[n] -= np.linalg.norm(v_i * value, axis=1) ** 2 / 2
                    res[n] += w_i * value
                res[n] += np.linalg.norm(vx, axis=1) ** 2 / 2
            return res

        if self.x.ndim == 4:
            self.x = self.x.reshape(x.shape[0], -1)
        if self.x.ndim != 2:
            raise ValueError("Invalid input", self.x)
        self.update_set.add('v')
        if 'v' not in self.param:
            self.param['v'] = Param(param=np.random.rand(self.output_size, self.rank, self.input_size) * 1e-8)
        self.update_set.add('w')
        if 'w' not in self.param:
            self.param['w'] = Param(param=np.random.rand(self.output_size, self.input_size) * 1e-8)
        self.update_set.add('b')
        if 'b' not in self.param:
            self.param['b'] = Param(param=np.random.rand(self.output_size) * 1e-8)

        return np.sum(
            np.sum(
                self.param['v'].param.reshape(1, self.output_size, self.rank, self.input_size)
                * self.x.reshape(self.x.shape[0], 1, 1, self.input_size),
                axis=3
            ) ** 2,
            axis=2
        ) / 2 - np.sum(
            (
                self.param['v'].param.reshape(1, self.output_size, self.rank, self.input_size)
                * self.x.reshape(self.x.shape[0], 1, 1, self.input_size)
            ) ** 2,
            axis=(2, 3)
        ) / 2 + np.dot(self.x, self.param['w'].param.T) + self.param['b'].param

    def backward(self, dout):

        if len(self.x_original_shape) == 3:
            self.update_set.clear()

            self.param['b'].grad += np.sum(dout, axis=0) * 1

            vx = np.zeros((self.output_size, self.rank))

            dx = np.zeros_like(self.x)

            for n in range(self.x.shape[0]):
                for i in range(self.x.shape[1]):

                    channel = self.x[n, i, 0]
                    index = self.x[n, i, 1]
                    value = self.x[n, i, 2]
                    v_key = 'v-{}-{}'.format(channel, index)
                    w_key = 'w-{}-{}'.format(channel, index)

                    vx += self.param[v_key].param * value

                    dx[n, i, 2] = self.param[w_key].param
                    for j in range(self.x.shape[1]):
                        dx[n, i, 2] += self.param[v_key].param.dot(self.param[v_key].param.T) * self.param[w_key].param

            for n in range(self.x.shape[0]):
                for i in range(self.x.shape[1]):
                    channel = self.x[n, i, 0]
                    index = self.x[n, i, 1]
                    value = self.x[n, i, 2]
                    v_key = 'v-{}-{}'.format(channel, index)
                    w_key = 'w-{}-{}'.format(channel, index)
                    self.param[w_key].grad += dout[n, :] * self.x[n, i, 2]
                    self.param[v_key].grad += value * dout[n, :] * (vx - value * self.param[v_key].param)

            return dx

        self.param['b'].grad += np.sum(dout, axis=0).T
        self.param['w'].grad += np.dot(self.x.T, dout).T
        self.param['v'].grad += np.sum(
            (
                np.sum(
                    self.param['v'].param.reshape(1, self.output_size, self.rank, self.input_size)
                    * self.x.reshape(self.x.shape[0], 1, 1, self.input_size),
                    axis=3
                ).reshape(
                    self.x.shape[0], self.output_size, self.rank, 1
                ) * self.x.reshape(
                    self.x.shape[0], 1, 1, self.input_size
                ) - self.param['v'].param.reshape(1, self.output_size, self.rank, self.input_size)
                * self.x.reshape(self.x.shape[0], 1, 1, self.input_size) ** 2
            ) * dout.reshape(self.x.shape[0], -1, 1, 1),
            axis=0
        )

        return np.sum(
            self.param['v'].param.reshape(
                1, self.output_size, self.rank, self.input_size
            ) ** 2 * self.x.reshape(
                self.x.shape[0], 1, 1, self.input_size
            ) / 2
            + self.param['w'].param.reshape(
                1, self.output_size, 1, self.input_size
            ),
            axis=(1, 2)
        ).reshape(*self.x_original_shape)


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
    def __init__(self, weight_shape, bias_shape, stride=1, pad=0, sparse=False, reg=Nothing()):
        Layer.__init__(self)
        State.__init__(self, {
            'w': Param(weight_shape, sparse=sparse, reg=reg),
            'b': Param(bias_shape, sparse=sparse)
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
