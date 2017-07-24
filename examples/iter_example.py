from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from oml.datasouces.iterator import BaseIterator, Page

import numpy as np


class Cifar10Iterator(BaseIterator):
    def __init__(self, data, batch_size=1):
        BaseIterator.__init__(self, item_to_value=self._item_to_value_func, max_results=len(data))
        self._data = data
        self.batch_size = batch_size

    @staticmethod
    def _item_to_value_func(data):
        return np.hstack(data)

    def _next_page(self):
        items = self._data[
                self.page_number * self.batch_size:min((self.page_number + 1) * self.batch_size, self.max_results)
                ]
        if len(items) == 0:
            return None
        page = Page(items, self._item_to_value)
        return page

    def initialize(self):
        self.page_number = 0
        self.num_results = 0
        self._started = False
