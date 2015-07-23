import numpy as np

def sigmoid(a):
    sigmoid = 1/(1+np.exp(-a))
    return sigmoid

def jacobian_sigmoid(activation_vector):
    size = activation_vector.shape[0]
    jacobian = np.zeros((size, size))
    jacobian[xrange(size), xrange(size)] += activation_vector*(1 - activation_vector)
    return jacobian
