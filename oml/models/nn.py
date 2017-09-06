from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from .components import Affine, Convolution, Pooling, Softmax, State, Relu, BatchNormalization, Dropout, FactorizationMachine
from .regularizers import L2Sq, Nothing
from .model import Classifier
from oml.functions import Differentiable

import numpy as np


def compute_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


class NN(Classifier, Differentiable):
    def __init__(self, input_shape=(1, 28, 28), last_layer=Softmax(), architecture=(
            {'layer': 'conv', 'kernel_num': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'layer': 'batch_normalization'},
            {'layer': 'activation', 'instance': Relu()},
            {'layer': 'pooling', 'pool_size': 2, 'stride': 2, 'padding': 0},
            {'layer': 'affine', 'unit_num': 50, 'reg': L2Sq(param=0.001)},
            {'layer': 'dropout'}
    )):
        Differentiable.__init__(self, gamma=1)
        self.input_shape = input_shape
        layers = []
        channel, height, width = input_shape
        hidden_size = channel * height * width
        for arch in architecture:
            if 'layer' in arch:
                if arch['layer'] == 'conv':
                    if isinstance(arch['kernel_size'], int):
                        kernel_size = (arch['kernel_size'], arch['kernel_size'])
                    elif isinstance(arch['kernel_size'], tuple) or isinstance(arch['kernel_size'], list):
                        kernel_size = arch['kernel_size']
                    else:
                        raise ValueError('Kernel size is invaid type', arch['kernel_size'])
                    layers.append(
                        Convolution(
                            (arch['kernel_num'], channel) + kernel_size,
                            (arch['kernel_num']),
                            arch['stride'],
                            arch['padding'],
                            reg=arch.get('reg', Nothing()),
                        )
                    )
                    channel = arch['kernel_num']
                    height = compute_output_size(height, kernel_size[0], arch['stride'], arch['padding'])
                    width = compute_output_size(width, kernel_size[1], arch['stride'], arch['padding'])
                    hidden_size = channel * height * width
                elif arch['layer'] == 'pooling':
                    if isinstance(arch['pool_size'], int):
                        pool_size = (arch['pool_size'], arch['pool_size'])
                    elif isinstance(arch['pool_size'], tuple) or isinstance(arch['pool_size'], list):
                        pool_size = arch['pool_size']
                    else:
                        raise ValueError('Pool size is invaid type', arch['pool_size'])
                    layers.append(
                        Pooling(pool_size, arch['stride'], arch['padding'])
                    )
                    height = compute_output_size(height, pool_size[0], arch['stride'], arch['padding'])
                    width = compute_output_size(width, pool_size[1], arch['stride'], arch['padding'])
                    hidden_size = channel * height * width
                elif arch['layer'] == 'activation':
                    layers.append(arch['instance'])
                elif arch['layer'] == 'affine':
                    layers.append(Affine(hidden_size, arch['unit_num'], reg=arch.get('reg', Nothing())))
                    channel = 1
                    height = 1
                    width = arch['unit_num']
                    hidden_size = channel * height * width
                elif arch['layer'] == 'batch_normalization':
                    layers.append(
                        BatchNormalization(
                            gamma_initializer=arch.get('gamma_initializer', np.ones),
                            beta_initializer=arch.get('beta_initializer', np.zeros),
                            momentum=arch.get('momentum', 0.9),
                            running_mean=arch.get('running_mean', None),
                            running_var=arch.get('running_var', None),

                        )
                    )
                elif arch['layer'] == 'dropout':
                    layers.append(
                        Dropout(arch.get('dropout_ratio', 0.5))
                    )
                else:
                    raise NotImplementedError('not registered architecture!')
            else:
                continue
        last_layer = last_layer

        Classifier.__init__(self, layers, last_layer)

    def predict(self, x: np.ndarray, *args, **kwargs):
        x = np.asarray(x).reshape((-1,) + self.input_shape)
        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
        return self.last_layer.predict(x, *args, **kwargs)

    def loss(self, x: np.ndarray, t: np.ndarray, *args, **kwargs):
        reg = 0
        t = np.asarray(t)
        if t.ndim == 1:
            batch_size = t.size
        else:
            batch_size = t.shape[0]
        x = np.asarray(x).reshape((batch_size,) + self.input_shape)
        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
            if isinstance(layer, State):
                for key in layer.param.keys():
                    reg += layer.param[key].reg.apply(layer.param[key].param)
        return self.last_layer.forward(x, t, *args, **kwargs) + reg
