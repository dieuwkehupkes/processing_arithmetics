from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt

network = SRN(3, 3, 3, 0.2)
network.generate_update_function()
network.generate_network_dynamics()


# test and training sequences
a = np.array([0,0,1])
b = np.array([0,1,0])
c = np.array([1,0,0])
# training_sequence = np.array([a, b, c, a, b, c, a, b, c])
training_sequence1 = np.array([[a, b, c, a, b, c, a, b, c], [c, a, b, c, a, b]])
training_sequence2 = np.array([[a, b, c, a, b, c, a, b, c, a], [b, a, c, b, a, c, b, a, c]])
test_sequence1 = np.array([a, b, c, a, b, c, a, b, c])
test_sequence2 = np.array([b, a, c, b, a, c, b, a, c])

rounds = np.arange(0, 1000, 1)
error_same = []
error_diff = []
error_rand = []

for round in rounds:
    network.train(training_sequence1, 1)

    error_same.append(network.compute_raining_error(test_sequence1))
    error_diff.append(network.compute_error(test_sequence2))

plt.plot(rounds, error_same, label="Same sequence")
plt.plot(rounds, error_diff, label="different sequence")
plt.legend(loc=2)
plt.xlabel("number of training rounds")
plt.ylabel("cross-entropy")
plt.show()
