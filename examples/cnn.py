from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.preprocessing import maxabs_scale
from oml.models.nn import NN
from oml.models.regulizers import L2Sq, L1
from oml.optimizers.adagrad import AdaRDA, AdaGrad
from oml.optimizers.rda import Rda
from oml.optimizers.fobos import Fobos
from oml.optimizers.vr import Svrg
from oml.optimizers.freerex import FreeRex
from oml.datasouces.iterator import NumpyIterator
from oml.models.components import Relu

from sklearn.datasets import fetch_mldata

from matplotlib import pyplot as plt
import os

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

architecture = [
    {'layer': 'conv', 'kernel_num': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'layer': 'activation', 'instance': Relu()},
    {'layer': 'pooling', 'pool_size': 2, 'stride': 2, 'padding': 0},
    {'layer': 'affine', 'unit_num': 150, 'reg': L1(param=0.001/1000)},
]


out = 'cnn_out_l1'


def opt_test(optimizer, label):
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
    if not os.path.isfile('./{}/{}_{}.csv'.format(out, label, 'loss')):
        print(label)
        optimizer.optimize(train_iter, test_iter, show_evaluation=True, show_loss=True, epoch=5)
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'loss'), optimizer.loss, delimiter=',')
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'evaluation'), optimizer.evaluation, delimiter=',')


train_iter = NumpyIterator(train_data, batch_size=100)
test_iter = NumpyIterator(test_data, batch_size=100)

opt_test(AdaGrad(NN(architecture=architecture)), 'AdaGrad')
opt_test(FreeRex(NN(architecture=architecture)), 'FreeRex')
opt_test(Svrg(NN(architecture=architecture)), 'SVRG')
opt_test(Fobos(NN(architecture=architecture)), 'FOBOS')
opt_test(Rda(NN(architecture=architecture)), 'RDA')


def plot():
    for i, title in enumerate(['loss', 'evaluation']):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        if title == 'loss':
            plt.ylim([0, 10])
        for method in ['AdaGrad', 'FreeRex', 'SVRG', 'AdaRDA', 'FOBOS', 'RDA']:
            r = np.loadtxt('./{}/{}_{}.csv'.format(out, method, title))
            r = r[::max(len(r) // 100, 1)]
            plt.plot(list(range(len(r))), r, label=method)
        plt.legend()
    plt.savefig('{}.png'.format(out))


plot()
