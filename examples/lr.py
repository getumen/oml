from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.datasets import load_boston
from sklearn.preprocessing import maxabs_scale
from oml.models.glm import LinearRegression
from oml.models.regulizers import L1
from oml.optimizers.fobos import Fobos
from oml.optimizers.adagrad import AdaGrad, PrimalDualAdaGrad
from oml.optimizers.rda import Rda
from oml.datasouces.iterator import NumpyIterator


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

train_iter = NumpyIterator(train_data, batch_size=3)
test_iter = NumpyIterator(test_data)


model1 = LinearRegression(feature, target, reg=L1(param=0.01))
optimizer = Fobos(model1)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model2 = LinearRegression(feature, target, reg=L1(param=0.01))
optimizer = AdaGrad(model2)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model3 = LinearRegression(feature, target, reg=L1(param=0.01))
optimizer = PrimalDualAdaGrad(model3)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)

model4 = LinearRegression(feature, target, reg=L1(param=0.01))
optimizer =Rda(model4)
optimizer.optimize(train_iter, test_iter, show_evaluation=True)
