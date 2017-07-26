from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np
import os

from sklearn.preprocessing import maxabs_scale
from oml.models.glm import SoftmaxRegression
from oml.models.regulizers import L2Sq, L1
from oml.optimizers.adagrad import AdaRDA, AdaGrad
from oml.optimizers.adam import AdMax, Adam
from oml.optimizers.rda import Rda, AcceleratedRDA
from oml.optimizers.fobos import Fobos
from oml.optimizers.vr import Svrg
from oml.optimizers.freerex import FreeRex
from oml.optimizers.smidas import Smidas
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
out = 'softmax_out'


def opt_test(optimizer, label):
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
    if not os.path.isfile('./{}/{}_{}.csv'.format(out, label, 'loss')):
        print(label)
        optimizer.optimize(train_iter, test_iter, show_evaluation=True)
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'loss'), optimizer.loss, delimiter=',')
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'rmse'), optimizer.evaluation, delimiter=',')

    results[label] = {
        'loss': optimizer.loss,
        'rmse': optimizer.evaluation
    }

opt_test(AdMax(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AdMax')
opt_test(Adam(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'Adam')
opt_test(FreeRex(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'FreeRex')
opt_test(AdaGrad(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AdaGrad')
opt_test(AdaRDA(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AdaRDA')
opt_test(Fobos(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'Fobos')
opt_test(Rda(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'RDA')
opt_test(AcceleratedRDA(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'AccRDA')
opt_test(Svrg(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'SVRG')
opt_test(Smidas(SoftmaxRegression(feature, target, reg=L1(0.0001))), 'Smidas')


def plot():
    for i, title in enumerate(['loss', 'rmse']):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        for method in ['AdaGrad', 'FreeRex', 'SVRG', 'Fobos', 'Adam', 'AdMax', 'RDA', 'AdaRDA', 'AccRDA', 'Smidas']:
            r = np.loadtxt('./{}/{}_{}.csv'.format(out, method, title))
            r = r[::max(len(r) // 100, 1)]
            plt.plot(list(range(len(r))), r, label=method)
        plt.legend()
    plt.savefig('{}.png'.format(out))


plot()
