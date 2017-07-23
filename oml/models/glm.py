from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.models.components import Affine, Gauss, Softmax, State, Poisson
from oml.models.regulizers import Nothing
from oml.models.model import Classifier, Regression

"""
Generalized Linear Models
"""


class LinearRegression(Regression):
    def __init__(self, input_size, output_size, reg=Nothing()
                 ):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Gauss(),
        )


class PoissonRegression(Regression):
    def __init__(self, input_size, output_size, reg=Nothing()):
        Regression.__init__(
            self,
            [Affine(input_size, output_size, reg=reg)],
            Poisson(),
        )


class SoftmaxRegression(Classifier):
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
