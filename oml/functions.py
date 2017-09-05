from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hinge(x):
    return np.maximum(0, x)


def softplus(x):
    return np.log(1 + np.exp(x))


def softmax(x):
    if x.shape[0] != x.size:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    return sigmoid(x)


def mean_squared(y, t):
    return 0.5 * np.linalg.norm(y - t) ** 2


def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        return -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / y.size
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t.astype(int)])) / batch_size


class StrongConvexity:
    def __init__(self, mu=0):
        # strong convexity parameter of the function
        self.mu = mu


class Differentiable:
    def __init__(self, gamma=None):
        # Lipschitz constant of the gradient of the function
        self.gamma = gamma
