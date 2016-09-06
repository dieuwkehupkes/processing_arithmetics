import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

damp_factor = 0.7
N = 20
n = 300

network_state = T.vector(dtype=theano.config.floatX)
W = T.matrix(dtype=theano.config.floatX)
W_back = T.matrix(dtype=theano.config.floatX)

samples = T.vector()

def next_training(output, network_state, W, W_back):
    return T.dot(network_state, W)
    next_state = T.tanh(T.dot(network_state, W) + T.dot(output, W_back))
    return next_state

states, _ = theano.scan(next_training, sequences=samples, non_sequences=[W, W_back], outputs_info=T.zeros_like(network_state))
# outs, _ = theano.scan(next, sequences=samples

sample_output = states[-200:]
sample_targ = samples[-200:]
# W_out = T.nlinalg.MatrixPinv(W)
# W_out = (T.nlinalg.MatrixPinv(sample_output), sample_targ)
# W_out = T.dot(T.nlinalg.MatrixPinv(sample_output), sample_targ)

givens = {network_state:    T.zeros_like(network_state)}

output_matrix = theano.function([samples, W, W_back], outputs=sample_output, givens=givens)
# output_matrix = theano.function([samples, W, W_back], outputs=W_out)

W = damp_factor*(0.5-np.random.random((N,N)))
W_back = damp_factor*(0.5-np.random.random(N))          # feed output back into network


samples = np.array([0.5*np.sin(float(x)/4) for x in xrange(n)])
W_out = output_matrix(samples, W, W_back)







