from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division


class FirstOrderOracle:
    def __init__(self):
        self.param = {}
        self.grad = {}


class OracleRef:
    def get_oracle(self) -> list:
        raise NotImplementedError()
