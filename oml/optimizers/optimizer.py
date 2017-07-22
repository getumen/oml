from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division


class Optimizer:
    def __init__(
            self,
            model,
            t=0,
    ):
        self.t = t

        self.model = model

    def optimize(self, train_data, test_data):
        raise NotImplementedError()
