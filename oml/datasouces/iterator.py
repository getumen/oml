from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Dict

import six


class Page:
    def __init__(self, items, item_to_value):
        self._items = items
        self._item_to_value = item_to_value
        self._num_items = len(items)
        self._remaining = self._num_items
        self._item_iter = iter(items)

    @property
    def num_items(self):
        return self._num_items

    @property
    def remaining(self):
        return self._remaining

    def __iter__(self):
        return self

    def next(self):
        item = six.next(self._item_iter)
        result = self._item_to_value(item)
        self._remaining -= 1
        return result

    __next__ = next


class BaseIterator:
    def __init__(self, item_to_value, max_results):
        self._started = False
        self._item_to_value = item_to_value
        self.max_results = max_results
        self.page_number = 0
        self.num_results = 0

    def _page_iter(self, increment):
        page = self._next_page()
        while page is not None:
            self.page_number += 1
            if increment:
                self.num_results += page.num_items
            yield page
            page = self._next_page()

    @property
    def pages(self):
        if self._started:
            raise ValueError('Iterator has already started', self)
        self._started = True
        return self._page_iter(increment=True)

    def _items_iter(self):
        for page in self._page_iter(increment=False):
            for item in page:
                self.num_results += 1
                yield item

    def __iter__(self):
        if self._started:
            raise ValueError('Iterator has already started', self)
        self._started = True
        return self._items_iter()

    def initialize(self):
        raise NotImplementedError()

    def _next_page(self):
        raise NotImplementedError


class NumpyIterator(BaseIterator):
    def __init__(self, data, batch_size=1):
        BaseIterator.__init__(self, item_to_value=self._item_to_value_func, max_results=data.shape[0])
        self._data = data
        self.batch_size = batch_size

    @staticmethod
    def _item_to_value_func(data):
        return data[:-1], data[-1]

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


class DictIterator(BaseIterator):
    def __init__(self, x: List[Dict[str, float]], t: List[float], batch_size=1):
        BaseIterator.__init__(self, item_to_value=self._item_to_value_func, max_results=len(x))
        self.x = x
        self.t = t
        self.batch_size = batch_size

    @staticmethod
    def _item_to_value_func(data):
        return data

    def _next_page(self):
        rng = slice(
            self.page_number * self.batch_size,
            min((self.page_number + 1) * self.batch_size, self.max_results)
        )
        items = list(zip(self.x[rng], self.t[rng]))
        if len(items) == 0:
            return None
        page = Page(items, self._item_to_value)
        return page

    def initialize(self):
        self.page_number = 0
        self.num_results = 0
        self._started = False
