"""
9/18/17
kasim
se.kasim.ebrahim@gmail.com
"""

import numpy as np


def func(g):
    return np.array([(np.exp(_g)) / (np.sum(np.exp(g))) for _g in g])
