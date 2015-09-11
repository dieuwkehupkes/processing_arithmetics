from SRN_Theano import SRN
import numpy as np

network = SRN(3, 3, 3, 0.2)
network.generate_update_function()
# network.generate_network_dynamics()


# test and training sequences
a = np.array([0,0,1])
b = np.array([0,1,0])
c = np.array([1,0,0])
training_sequence = np.array([[a, b, c, a, b, c, a, b, c, a], [b, a, c, b, a, c, b, a, c]])
test_sequence1 = np.array([a, b, c, a, b, c, a, b, c])
test_sequence2 = np.array([b, a, c, b, a, c, b, a, c])

print network.activations['hidden_t']
print network.activations['output_t']

network.forward_pass(a)
