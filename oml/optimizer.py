from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.datasouces.iterator import BaseIterator


class Optimizer:
    def __init__(
            self,
            model,
            train_data: BaseIterator,
            test_data: BaseIterator,
            t=0,
            verbose=True,
    ):
        self.t = t

        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.verbose = verbose

    def optimize(self):
        raise NotImplementedError()
