from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.preprocessing import maxabs_scale
from oml.models.glm import SoftmaxRegression
from oml.models.regulizers import L2Sq, L1
from oml.optimizers.adagrad import PrimalDualAdaGrad, AdaGrad
from oml.optimizers.rda import Rda
from oml.optimizers.fobos import Fobos
from oml.optimizers.vr import Svrg
from oml.optimizers.freerex import FreeRex
from oml.datasouces.iterator import NumpyIterator

from sklearn.datasets import fetch_mldata

from matplotlib import pyplot as plt

mnist = fetch_mldata('MNIST original')

x = maxabs_scale(mnist['data'])
t = mnist['target']

feature = x.shape[1]
target = 10

data = np.hstack((x, t.reshape(-1, 1)))

np.random.shuffle(data)

train_index = np.random.choice(range(data.shape[0]), data.shape[0] // 7 * 6, replace=False)
test_index = list(set(range(data.shape[0])).difference(set(train_index)))

train_data = data[train_index, :]
test_data = data[test_index, :]


train_iter = NumpyIterator(train_data, batch_size=100)
test_iter = NumpyIterator(test_data, batch_size=100)

results = {}


def opt_test(optimizer, label):
    print(label)
    optimizer.optimize(train_iter, test_iter, show_evaluation=True)

    results[label] = {
        'loss': optimizer.loss,
        'evaluation': optimizer.evaluation
    }

opt_test(FreeRex(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'FreeRex')
opt_test(AdaGrad(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AdaGrad')
opt_test(PrimalDualAdaGrad(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AdaRDA')
opt_test(Fobos(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'FOBOS')
opt_test(Rda(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'RDA')
opt_test(Svrg(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'SVRG')


def plot(result):
    for i, title in enumerate(['loss', 'evaluation']):
        plt.subplot(1, 2, i+1)
        plt.title(title)
        for method in result.keys():
            plt.plot(list(range(len(result[method][title]))), result[method][title], label=method)
        plt.legend()
    plt.savefig('softmax.png')

plot(results)
