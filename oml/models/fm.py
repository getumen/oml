from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from oml.functions import Differentiable
from oml.models.components import FactorizationMachine, Gauss, Poisson
from oml.models.model import Regression
from oml.models.regularizers import Nothing
from oml.models.components import State

import numpy as np

from typing import List, Dict


class BaseFM(Regression):

    def __init__(self, layer, last_layer):
        Regression.__init__(self, layer, last_layer)

    def evaluate_model(self, test_iter, show=False):
        error = 0
        sample_num = 0
        for page in test_iter.pages:
            x, t = zip(*list(page))
            t = np.asarray(t)
            y = self.predict(x, train_flg=False).reshape(len(t))
            error += np.linalg.norm(t - y) ** 2
            sample_num += len(x)
        if show:
            print('=== RMSE: {}'.format(np.sqrt(error / sample_num)))
        return np.sqrt(error / sample_num)

    def predict(self, x: List[Dict[str, float]], *args, **kwargs):

        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
        return self.last_layer.predict(x, *args, **kwargs)

    def loss(self, x: List[Dict[str, float]], t, *args, **kwargs):
        reg = 0

        t = np.asarray(t)

        for layer in self.layers:
            x = layer.forward(x, *args, **kwargs)
            if isinstance(layer, State):
                for key in layer.param.keys():
                    reg += layer.param[key].reg.apply(layer.param[key].param)
        return self.last_layer.forward(x, t, *args, **kwargs) + reg


class FM(BaseFM, Differentiable):
    def __init__(
            self,
            rank_list=(1, 5),
            reg=Nothing()
    ):
        BaseFM.__init__(
            self,
            [FactorizationMachine(rank_list=rank_list, reg=reg)],
            Gauss(),
        )
        Differentiable.__init__(self, gamma=1)


class PoissonFM(BaseFM, Differentiable):
    def __init__(
            self,
            rank_list=(1, 5),
            reg=Nothing()
    ):
        BaseFM.__init__(
            self,
            [FactorizationMachine(reg=reg, rank_list=rank_list)],
            Poisson(),
        )
        Differentiable.__init__(self, gamma=1)

