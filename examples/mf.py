from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from matplotlib import pyplot as plt

from oml.datasouces.iterator import NumpyIterator
from oml.models.fm import FM
from oml.models.regulizers import L1, L2Sq
from oml.optimizers.adagrad import AdaGrad
from oml.optimizers.fobos import Fobos
from oml.optimizers.freerex import FreeRex
from oml.optimizers.vr import Svrg
from oml.optimizers.adam import Adam, AdMax
from oml.optimizers.nesterov import AccSGD

data = np.loadtxt('./ml-latest-small/ratings.csv', skiprows=1, delimiter=',')

np.random.shuffle(data)

data = data[:, :3].astype(int)

train_iter = NumpyIterator(data[:data.shape[0] // 5 * 4], batch_size=100)
test_iter = NumpyIterator(data[data.shape[0] // 5 * 4:], batch_size=1000)

results = {}

out = 'fm_out'


def opt_test(optimizer, label):
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
    if not os.path.isfile('./{}/{}_{}.csv'.format(out, label, 'loss')):
        print(label)
        optimizer.optimize(train_iter, test_iter, show_evaluation=True, show_loss=True, epoch=5)
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'loss'), optimizer.loss, delimiter=',')
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'rmse'), optimizer.evaluation, delimiter=',')

    results[label] = {
        'loss': optimizer.loss,
        'rmse': optimizer.evaluation
    }


opt_test(FreeRex(FM(input_bias_reg=L1(), variance_reg=L2Sq())), 'FreeRex')
opt_test(AdaGrad(FM(input_bias_reg=L1(), variance_reg=L2Sq()), step_size=0.1), 'AdaGrad')
opt_test(Fobos(FM(input_bias_reg=L1(), variance_reg=L2Sq())), 'Fobos')
opt_test(Svrg(FM(input_bias_reg=L1(), variance_reg=L2Sq())), 'SVRG')
opt_test(Adam(FM(input_bias_reg=L1(), variance_reg=L2Sq())), 'Adam')
opt_test(AdMax(FM(input_bias_reg=L1(), variance_reg=L2Sq())), 'AdMax')


def plot():
    for i, title in enumerate(['loss', 'rmse']):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        for method in ['AdaGrad', 'FreeRex', 'SVRG', 'Fobos', 'Adam', 'AdMax']:
            r = np.loadtxt('./{}/{}_{}.csv'.format(out, method, title))
            r = r[::max(len(r) // 100, 1)]
            plt.plot(list(range(len(r))), r, label=method)
        plt.legend()
    plt.savefig('{}.png'.format(out))


plot()
