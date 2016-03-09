from SRN_Theano import *

x = SRN(3, 3, 3, 0.2)
x.generate_network_dynamics()

# training sequence
a = np.array([0,0,1])
b = np.array([0,1,0])
c = np.array([1,0,0])

training_sequence = np.array([[a, b, c, a, b, c, a, b, c, a], [b, a, c, b, a, c, b, a, c]])
test_sequence1 = np.array([a, b, c, a, b, c, a, b, c])
test_sequence2 = np.array([b, a, c, b, a, c, b, a, c])

x.train(training_sequence, 1, 0)
