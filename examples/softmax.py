from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.preprocessing import maxabs_scale
from oml.models.glm import SoftmaxRegression
from oml.models.regulizers import L2Sq, L1, PositiveBox
from oml.optimizers.adagrad import PrimalDualAdaGrad, AdaGrad
from oml.optimizers.rda import Rda
from oml.optimizers.fobos import Fobos
from oml.datasouces.iterator import NumpyIterator

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

x = maxabs_scale(mnist['data'])
t = mnist['target']

feature = x.shape[1]
target = 10

data = np.hstack((x, t.reshape(-1, 1)))

np.random.shuffle(data)

train_index = np.random.choice(range(data.shape[0]), data.shape[0] // 2, replace=False)
test_index = list(set(range(data.shape[0])).difference(set(train_index)))

train_data = data[train_index, :]
test_data = data[test_index, :]


train_iter = NumpyIterator(train_data, batch_size=100)
test_iter = NumpyIterator(test_data, batch_size=len(test_index))

model1 = SoftmaxRegression(feature, target, reg=L1(0.01/np.sqrt(x.shape[1])))
optimizer = PrimalDualAdaGrad(model1)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model2 = SoftmaxRegression(feature, target, reg=L2Sq(param=0.01))
optimizer = AdaGrad(model2)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model3 = SoftmaxRegression(feature, target, reg=L1(param=0.01/np.sqrt(x.shape[1])))
optimizer = Rda(model3)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model4 = SoftmaxRegression(feature, target, reg=L2Sq(param=0.01))
optimizer = Fobos(model4)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)
