from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt

network = SRN(4, 8, 4, 0.2)
network.generate_update_function()
network.generate_network_dynamics()


# test and training sequences
a = np.array([0,0,0,1])
b = np.array([0,0,1,0])
c = np.array([0,1,0,0])
d = np.array([1,0,0,0])
sequence_dict = dict(zip([1, 2, 3, 4], [a, b, c, d]))

# training_sequence = np.array([a, b, c, a, b, c, a, b, c])
training_sequence = np.array([[a, b, c, d, a, b, c, d, a, b, c, d, a], [b, a, d, c, b, a, d, c, b, a, d, c]])
test_sequence1 = np.array([a, b, c, d, a, b, c, d, a, b, c, d])
test_sequence2 = np.array([b, a, d, c, b, a, d, c, b, a, d, c])
test_sequence3 = np.array([b, b, b, b, b, b, b, b, b])

rounds = np.arange(0, 1500, 1)
error_same = []
error_diff = []
error_rand = []

for round in rounds:
    network.train(training_sequence, 1)

    rand_sequence = [sequence_dict[i] for i in np.random.randint(1, 5, size=12)]

    error_same.append(network.compute_error(test_sequence1))
    error_diff.append(network.compute_error(test_sequence2))
    error_rand.append(network.compute_error(rand_sequence))

plt.plot(rounds, error_same, label="abcd")
plt.plot(rounds, error_diff, label="badc")
plt.plot(rounds, error_rand, label="random sequence")
plt.legend(loc=2)
plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("cross-entropy")
plt.show()
