from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from matplotlib import pyplot as plt

from oml.datasouces.iterator import DictIterator
from oml.models.fm import PoissonFM
from oml.optimizers.adagrad import AdaGrad
from oml.optimizers.adam import Adam, AdMax
from oml.optimizers.sgd import Fobos

from oml.models.regularizers import L2Sq

data = np.loadtxt('./ml-latest-small/ratings.csv', skiprows=1, delimiter=',')

np.random.shuffle(data)

data = data[:, :3].astype(int)

x = []
t = []

for line in data:
    x.append({'u_{}'.format(line[0]): 1, 'i_{}'.format(line[1]): 1})
    t.append(line[2])

train_iter = DictIterator(x=x[:data.shape[0] // 5 * 4], t=t[:data.shape[0] // 5 * 4], batch_size=100)
test_iter = DictIterator(x=x[data.shape[0] // 5 * 4:], t=t[data.shape[0] // 5 * 4:], batch_size=1000)

results = {}

out = 'poisson_fm_out'


def opt_test(optimizer, label):
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
    if not os.path.isfile('./{}/{}_{}.csv'.format(out, label, 'loss')):
        print(label)
        optimizer.optimize(train_iter, test_iter, show_evaluation=True, epoch=5)
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'loss'), optimizer.loss, delimiter=',')
        np.savetxt('./{}/{}_{}.csv'.format(out, label, 'rmse'), optimizer.evaluation, delimiter=',')

    results[label] = {
        'loss': optimizer.loss,
        'rmse': optimizer.evaluation
    }


opt_test(Fobos(PoissonFM(reg=L2Sq())), 'Fobos')


def plot():
    for i, title in enumerate(['loss', 'rmse']):
        plt.subplot(1, 2, i + 1)
        plt.title(title)
        for method in results.keys():
            r = np.loadtxt('./{}/{}_{}.csv'.format(out, method, title))
            r = r[::max(len(r) // 100, 1)]
            plt.plot(list(range(len(r))), r, label=method)
        plt.legend()


plot()
plt.savefig('{}.png'.format(out))
