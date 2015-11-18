# Test example taken from backprop book

from SRN_Theano import *
import numpy as np

np.random.seed(0)

def generate_training_sequence(length):
    seq = []
    for i in xrange(length):
        x1, x2 = np.random.randint(2), np.random.randint(2)
        x3 = np.logical_xor(x1, x2)
        seq.append([[x1],[x2],[x3]])
    return np.array(seq)

def compute_training_error(sequence, network):
    for index in xrange(0, len(sequence) - 1):
        network.forward_pass(sequence[index])
        print "network output: ", network.activations['output_t'].get_value()
        print 'target output: ', sequence[index+1]

network = SRN(3, 3, 0.1)
network.generate_network_dynamics()
network.test_single_sequence()
network.generate_update_function()

# training sequence
a = np.array([0,0,1]).astype(theano.config.floatX)
b = np.array([0,1,0]).astype(theano.config.floatX)
c = np.array([1,0,0]).astype(theano.config.floatX)
seq1 = np.array([a, b, c, a, b, c, a, b, c])
seq2 = np.array([b, a, c, b, a, c, b, a, c])
seq1 = np.array([a,a,a,a,a,a])
seq2 = np.array([b,b,b,b,b,b])
training_sequence = np.array([seq1, seq2])

print "\ntraining error before training:", network.compute_error(seq1)
print network.compute_error(seq1)
print network.compute_error(seq2)

print "\nwhat happens? sequence 1:"
compute_training_error(seq1, network)

print "\nwhat happens? sequence 2:"
compute_training_error(seq2, network)

"""
print "\nNetwork weights:"
print "W:", network.W.get_value()
print "V:", network.V.get_value()
print "U:", network.U.get_value()
"""

raw_input()

network.train(training_sequence, 200, 1)

print "\ntraining error after 1 round of training:"
print network.compute_error(seq1)
print network.compute_error(seq2)

print "\nwhat happens? sequence 1:"
compute_training_error(seq1, network)

print "\nwhat happens? sequence 2:"
compute_training_error(seq2, network)

"""
print "\nNetwork weights:"
print "W:", network.W.get_value()
print "V:", network.V.get_value()
print "U:", network.U.get_value()
"""

raw_input()

network.train(training_sequence, 200, 1)

print "\ntraining error after 2 rounds of training:"
print network.compute_error(seq1)
print network.compute_error(seq2)

"""
print "\nNetwork weights:"
print "W:", network.W.get_value()
print "V:", network.V.get_value()
print "U:", network.U.get_value()
"""

raw_input()

network.train(training_sequence, 200, 1)

print "\ntraining error after 3 rounds of training:"
print network.compute_error(seq1)
print network.compute_error(seq2)

print "\nwhat happens? sequence 2:"
compute_training_error(seq1, network)

print "\nwhat happens? sequence 2:"
compute_training_error(seq2, network)

"""
print "\nNetwork weights:"
print "W:", network.W.get_value()
print "V:", network.V.get_value()
print "U:", network.U.get_value()
"""
