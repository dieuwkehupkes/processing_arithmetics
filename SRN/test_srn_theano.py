from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt

network = SRN(10, 8, 10, 0.2)
network.generate_update_function()
network.generate_network_dynamics()


# test and training sequences
lexicon = np.zeros((10,10), float)
np.fill_diagonal(lexicon, 1)
a, b, c, d, e, f, g, h, i, j = lexicon

training_sequence1 = np.array([[a, b, c, d, a, b, c, d, a], [b, a, d, c, b, a, d, c, b]])
training_sequence2 = np.array([[a, b, c, d, a, b, c, d, a], [f, b, c, i, f, b, c, i, f]])
training_sequence3 = np.array([[f, b, c, i, f, b, c, i, f], [a, b, c, d, a, b, c, d, a]])
test_sequence1 = np.array([a, b, c, d, a, b, c, d, a, b, c, d, a])
test_sequence2 = np.array([b, a, d, c, b, a, d, c, b, a, d, c, b])
test_sequence3 = np.array([f, b, c, i, f, b, c, i, f, b, c, i, f])

stepsize = 1
rounds = np.arange(0, 3500, stepsize)
error1, error2, error_rand = [], [], []
prediction1, prediction2, prediction_rand = [], [], []

for round in rounds:
    network.train(training_sequence2, stepsize)

    rand_sequence = [lexicon[i] for i in np.random.randint(0, 10, size=12)]

    error1.append(network.compute_error(test_sequence1))
    error2.append(network.compute_error(test_sequence3))
    error_rand.append(network.compute_error(rand_sequence))
    prediction1.append(network.compute_prediction_error(test_sequence1))
    prediction2.append(network.compute_prediction_error(test_sequence3))
    prediction_rand.append(network.compute_prediction_error(rand_sequence))

print "cross entropy sequence 1:", error1[-1]
print "cross entropy sequence 2:", error2[-1]

print "prediction rate sequence 1:", prediction1[-1]
print "prediction rate sequence 2:", prediction2[-1]

plt.plot(rounds, error1, label="abcd, cross-entropy")
plt.plot(rounds, error2, label="fbci, cross-entropy")
plt.plot(rounds, error_rand, label="random sequence, cross_entropy")
plt.plot(rounds, prediction1, label="abcd, prediction error")
plt.plot(rounds, prediction2, label="fbci, prediction error")
plt.plot(rounds, prediction_rand, label="random sequence, prediction error")
plt.legend(loc=2)
plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("cross-entropy/prediction error")
plt.show()
