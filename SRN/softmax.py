import numpy as np


def softmax(a):
    e = np.exp(a)
    marg = e.sum()
    return e/marg

def jacobian_softmax(activation_vector):
    size = activation_vector.shape[0]
    jacobian = - np.outer(activation_vector, activation_vector)
    jacobian[xrange(size), xrange(size)] += activation_vector 

    return jacobian


