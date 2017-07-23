from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import generators
from __future__ import division

import numpy as np


class Reg:
    def __init__(self, param=0.01):
        self.param = param

    def apply(self, w):
        raise NotImplementedError()

    def proximal(self, w, step_size):
        raise NotImplementedError()

    def sub_differential(self, w):
        return w - self.proximal(w, 1)


class FineteLipschitz:
    pass


class Nothing(Reg):
    def __init__(self):
        Reg.__init__(self)

    def apply(self, w):
        return 0

    def proximal(self, w, step_size):
        return w


class L2Sq(Reg, FineteLipschitz):
    """
    L2 squared norm does not have finite Lipschitz constant in general.
    However, if we do not have neither Lipschitz constant nor bounded parameter space,
    all online learning algorithms attain exponential regret!
    So we assume parameter space is bounded or objective has finite Lipchitz constant.
    """

    def __init__(self, param=0.01):
        Reg.__init__(self, param)

    def apply(self, w):
        return self.param * np.linalg.norm(w) ** 2 / 2

    def proximal(self, w, step_size):
        return (1 - self.param * step_size) * w


class L1(Reg, FineteLipschitz):
    """
    Tibshirani, Robert.
    "Regression shrinkage and selection via the lasso."
    Journal of the Royal Statistical Society. Series B (Methodological) (1996): 267-288.
    """

    def __init__(self, param=0.01):
        Reg.__init__(self, param)

    def apply(self, w):
        return self.param * np.sum(np.absolute(w))

    def proximal(self, w, step_size):
        return np.sign(w) * np.maximum(np.absolute(w) - self.param * step_size, 0)


class Trace(Reg, FineteLipschitz):
    """
    computational complexity is $O(d^3)$!
    """

    def __init__(self, param=0.01):
        Reg.__init__(self, param)

    def apply(self, w):
        if w.ndim != 2:
            raise ValueError("Dimension must be 2!")
        return self.param * np.trace(w.T.dot(w))

    def proximal(self, w, step_size):
        U, s, V = np.linalg.svd(w, full_matrices=False)
        s = np.sign(s) * np.maximum(np.absolute(s) - self.param * step_size, 0)
        return U.dot(np.diag(s)).dot(V)


class PositiveBox(Reg):
    def __init__(self):
        Reg.__init__(self)

    def apply(self, w):
        return np.inf if np.any(w < 0) else 0

    def proximal(self, w, step_size):
        return np.maximum(w, 0)


class ProximalAverage:
    """
    Yu, Yao-Liang.
    "Better approximation and faster algorithm using the proximal average."
    Advances in neural information processing systems (2013): 458-466.

    Reg must have Lipschitz constant!
    """

    def __init__(self, reg_list=(L2Sq(),), weight=None):
        self.size = len(reg_list)
        self.reg_list = reg_list
        for reg in reg_list:
            if not isinstance(reg, Reg):
                raise ValueError("Contain invalid regularization term!")
            if not isinstance(reg, FineteLipschitz):
                raise ValueError("""
                Proximal Average not work with infinite Lipschitz regularizers
                Use: 
                Yurtsever, Alp, Bang Công Vu, and Volkan Cevher. 
                "Stochastic three-composite convex minimization." 
                Advances in Neural Information Processing Systems. 2016.
                """)
        if weight is None:
            self.weight = np.ones_like(self.reg_list) / self.size
        else:
            self.weight = weight / np.sum(weight)

    def apply(self, w):
        return np.average([f.apply(w) for f in self.reg_list], weights=self.weight)

    def proximal(self, w, step_size):
        return np.average([f.proximal(w, step_size / p) for f, p in zip(self.reg_list, self.weight)])
