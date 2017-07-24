from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.datasets import load_boston
from sklearn.preprocessing import maxabs_scale
from oml.models.glm import LinearRegression
from oml.models.regulizers import L1, L2Sq
from oml.optimizers.fobos import Fobos
from oml.optimizers.adagrad import AdaGrad, PrimalDualAdaGrad
from oml.optimizers.rda import Rda
from oml.optimizers.vr import Svrg
from oml.optimizers.freerex import FreeRex
from oml.datasouces.iterator import NumpyIterator

from matplotlib import pyplot as plt

data = load_boston()

x = maxabs_scale(data['data'])
t = data['target']

feature = x.shape[1]
target = 1

data = np.hstack((x, t.reshape(-1, 1)))

train_index = np.random.choice(range(data.shape[0]), data.shape[0] // 2, replace=False)
test_index = list(set(range(data.shape[0])).difference(set(train_index)))

train_data = data[train_index, :]
test_data = data[test_index, :]

train_iter = NumpyIterator(train_data, batch_size=10)
test_iter = NumpyIterator(test_data)

results = {}


def opt_test(optimizer, label):
    print(label)
    optimizer.optimize(train_iter, test_iter, show_evaluation=True, epoch=1000)

    results[label] = {
        'loss': optimizer.loss,
        'evaluation': optimizer.evaluation
    }

opt_test(FreeRex(LinearRegression(feature, target, reg=L2Sq(0.01))), 'FreeRex')
opt_test(AdaGrad(LinearRegression(feature, target, reg=L2Sq(0.01))), 'AdaGrad')
opt_test(PrimalDualAdaGrad(LinearRegression(feature, target, reg=L2Sq(0.01))), 'AdaRDA')
opt_test(Fobos(LinearRegression(feature, target, reg=L2Sq(0.01))), 'FOBOS')
opt_test(Rda(LinearRegression(feature, target, reg=L2Sq(0.01))), 'RDA')
opt_test(Svrg(LinearRegression(feature, target, reg=L2Sq(0.01))), 'SVRG')


def plot(result):
    for i, title in enumerate(['loss', 'evaluation']):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        for method in result.keys():
            plt.plot(list(range(len(result[method][title]))), result[method][title], label=method)
        plt.legend()
    plt.savefig('lr.png')


plot(results)
