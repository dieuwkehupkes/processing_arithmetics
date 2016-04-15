"""
Collection of auxiliary functions used in different
classes.
"""


import numpy as np
import theano.tensor as T


def piecewise_linear(input_vector):
    """
    For every element in input_vector, return
    f(net)  = net   iff 0 <= net <= 1
            = 0     iff net < 0 
            = 1     iff net > 1
    """
    clipped = T.clip(input_vector, 0, 1)
    return clipped


def softmax_tensor(input_tensor):
    """
    Softmax function that can be applied to a 
    three dimensional tensor.
    """
    d0, d1, d2 = input_tensor.shape
    reshaped = T.reshape(input_tensor, (d0*d1, d2))
    softmax_reshaped = T.nnet.softmax(reshaped)
    softmax = T.reshape(softmax_reshaped, newshape=input_tensor.shape)
    return softmax
