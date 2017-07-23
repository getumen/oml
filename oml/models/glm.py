from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.models.components import Affine, Gauss, Softmax, State, Poisson
from oml.models.regulizers import Nothing
from oml.models.model import Classifier, Regression
from oml.functions import Differentiable

import numpy as np

"""
Generalized Linear Models
"""


class LinearRegression(Regression, Differentiable):
    def __init__(self, input_size, output_size, reg=Nothing()
                 ):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Gauss(),
        )
        Differentiable.__init__(self, gamma=1)


class PoissonRegression(Regression, Differentiable):
    def __init__(self, input_size, output_size, reg=Nothing(), domain_rad=1):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Poisson(),
        )
        Differentiable.__init__(self, gamma=np.exp(domain_rad))


class SoftmaxRegression(Classifier, Differentiable):
    def __init__(
            self,
            input_size,
            output_size,
            reg=Nothing()
    ):
        Classifier.__init__(
            self,
            [
                Affine(input_size, output_size, reg=reg)
            ],
            Softmax(),
        )
        Differentiable.__init__(self, gamma=1)
