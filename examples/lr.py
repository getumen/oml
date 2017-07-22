from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np

from sklearn.datasets import load_boston
from sklearn.preprocessing import maxabs_scale
from oml.models.glm import LinearRegression
from oml.optimizers.fobos import Fobos
from oml.datasouces.iterator import NumpyIterator


data = load_boston()

x = maxabs_scale(data['data'])
t = maxabs_scale(data['target'])

train_index = np.random.choice(range(x.shape[0]), x.shape[0]//2, replace=False)
test_index = list(set(range(x.shape[0])).difference(set(train_index)))


train_x = x[train_index, :]
test_x = x[test_index, :]

train_t = t[train_index]
test_t = t[test_index]

train_iter = NumpyIterator(train_x, train_t)
test_iter = NumpyIterator(test_x, test_t)

model = LinearRegression(x.shape[1], 1)

optimizer = Fobos(model)
optimizer.optimize(train_iter, test_iter)

