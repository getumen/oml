from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

from six.moves import cPickle


def get_binary(obj):
    return cPickle.dumps(obj)


def get_obj(binary):
    return cPickle.loads(binary)
