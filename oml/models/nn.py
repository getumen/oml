from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.models.components import Affine, Convolution, Pooling, Softmax, State
from oml.models.regulizers import Nothing
import numpy as np


# class NN:
#     def __init__(self, input_channel, last_layer, architecture=[
#         {}
#     ]):
#         self.layers = []
#         for arch in architecture:
#             if 'layer' in arch:
#                 if arch['layer'] == 'conv':
#                     self.layers.append(
#                         Convolution(
#                             (arch['kernel_num'], input_channel, arch['kernel_size'], arch['kernel_size']),
#                             (arch['kernel_num']),
#                             arch['stride'],
#                             arch['padding']
#                         )
#                     )
#                     input_channel = arch['kernel_num']
#             else:
#                 continue
#         self.last_layer = last_layer
#
#     def predict(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return self.last_layer.predict(x)
#
#     def loss(self, x, t):
#         reg = 0
#         for layer in self.layers:
#             x = layer.forward(x)
#             if isinstance(layer, State):
#                 for key in layer.param.keys():
#                     reg += layer.param[key].reg.apply(layer.param[key].param)
#         return self.last_layer.forward(x, t) + reg
#
#     def clear_grad(self):
#         for layer in self.layers:
#             if isinstance(layer, State):
#                 for value in layer.param.values():
#                     value.clear_grad()
#
#     def compute_grad(self):
#         dout = self.last_layer.backward()
#         for layer in reversed(self.layers):
#             dout = layer.backward(dout)
#
#     def evaluate_model(self, test_iter):
#         raise NotImplementedError()
