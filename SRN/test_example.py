# Test example taken from backprop book

from SRN import *
import numpy as np

def generate_training_sequence(length):
    seq = []
    for i in xrange(length):
        x1, x2 = np.random.randint(2), np.random.randint(2)
        x3 = np.logical_xor(x1, x2)
        seq.append([[x1],[x2],[x3]])
    return np.array(seq)

def compute_training_error(sequence, network):
    sse = 0.0
    network.update(sequence[0])
    for index in xrange(1, len(sequence) - 1):
        network.update(sequence[index])
        # print "network output: ", network.O
        # print 'target output: ', sequence[index+1]
        error = np.sqrt(np.power(sequence[index+1] - network.O, 2).sum())
        # print "error: ", error
        sse += error
        # raw_input()
    return sse/(len(sequence)-1)

network = SimpleRecurrentNetwork(3, 3, 3)
network.initialise_weights(-0.2, 0.2)

# training sequence
a = np.array([0,0,1])
b = np.array([0,1,0])
c = np.array([1,0,0])
training_sequence = np.array([[a, b, c, a, b, c, a, b, c, a], [b, a, c, b, a, c, b, a, c]])
test_sequence1 = np.array([a, b, c, a, b, c, a, b, c])
test_sequence2 = np.array([b, a, c, b, a, c, b, a, c])

print compute_training_error(test_sequence1, network)

# construct xor training sequence
# training_sequence = generate_training_sequence(1000)
network.train(training_sequence, learning_rate = 0.1, rounds = 1000, depth = 3)

test_sequence = np.array([[0], [0], [0], [0], [1], [1], [1], [0], [1], [1], [1], [0]])
print compute_training_error(test_sequence1, network)
print compute_training_error(test_sequence2, network)

network.train(training_sequence, learning_rate = 0.1, rounds = 1000, depth = 3)

print compute_training_error(test_sequence1, network)
print compute_training_error(test_sequence2, network)

network.train(training_sequence, learning_rate = 0.1, rounds = 1000, depth = 3)

print compute_training_error(test_sequence1, network)
print compute_training_error(test_sequence2, network)
