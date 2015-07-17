import numpy as np


def softmax(a):
    e = np.exp(a)
    marg = e.sum()
    return e/marg
