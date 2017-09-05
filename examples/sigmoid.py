from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from oml.datasouces.iterator import NumpyIterator
from oml.models.glm import SoftmaxRegression
from oml.models.regularizers import L2Sq
from oml.optimizers.freerex import FreeRex

if __name__ == '__main__':
    t = np.random.randint(2, size=(10000, 1))

    train_iter = NumpyIterator(
        np.hstack(
            (
                np.hstack((np.random.normal(size=(10000, 10)), t)), t
            )
        ),
        batch_size=20
    )

    t = np.random.randint(2, size=(10000, 1))

    test_iter = NumpyIterator(
        np.hstack(
            (
                np.hstack((np.random.normal(size=(10000, 10)), t)), t
            )
        ),
        batch_size=100
    )

    opt = FreeRex(SoftmaxRegression(input_size=11, output_size=1, reg=L2Sq()))
    opt.optimize(train_iter, test_iter, show_evaluation=True)
