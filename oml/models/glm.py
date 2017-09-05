from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.models.components import Affine, Gauss, Softmax, State, Poisson
from oml.models.regularizers import Nothing
from oml.models.model import Classifier, Regression
from oml.functions import Differentiable, StrongConvexity

import numpy as np

"""
Generalized Linear Models
"""


class LinearRegression(Regression, Differentiable, StrongConvexity):
    def __init__(self, input_size, output_size, reg=Nothing()
                 ):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Gauss(),
        )
        Differentiable.__init__(self, gamma=1)
        StrongConvexity.__init__(self, mu=1)


class PoissonRegression(Regression, Differentiable, StrongConvexity):
    def __init__(self, input_size, output_size, reg=Nothing(), domain_rad=1):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Poisson(),
        )
        Differentiable.__init__(self, gamma=np.exp(domain_rad))
        StrongConvexity.__init__(self, mu=0)


class SoftmaxRegression(Classifier, Differentiable, StrongConvexity):
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
        StrongConvexity.__init__(self, mu=0)
