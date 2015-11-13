from SRN_Theano_batch import SRN
import numpy as np
import matplotlib.pyplot as plt
import theano

network = SRN(10, 8, 0.2)
network.generate_update_function()
network.generate_network_dynamics()
network.test_single_sequence()

# test and training sequences
lexicon = np.zeros((10,10)).astype(theano.config.floatX)
np.fill_diagonal(lexicon, 1)
a, b, c, d, e, f, g, h, i, j = lexicon

seq1, l1 = np.array([a,b,d]), 'abd'
seq2, l2 = np.array([f,b,i]), 'fbi'
seq3, l3 = np.array([a, b, b, d]), 'abbd'
seq4, l4 = np.array([f, b, b, i]), 'fbbi'
seq5, l5 = np.array([a, b, b, b, d]), 'abbbd'
seq6, l6 = np.array([f, b, b, b, i]), 'fbbbi'
seq7, l7 = np.array([a, b, b, b, b, d]), 'abbbbd'
seq8, l8 = np.array([f, b, b, b, b, i]), 'fbbbbi'


training_sequence1 = np.array([seq1, seq2])
training_sequence2 = np.array([seq3, seq4])
training_sequence3 = np.array([seq5, seq6])
training_sequence4 = np.array([seq7, seq8])
# training_sequence2 = np.array([[a, b, c, d], [f, b, c, i]])

training_options = {'1': (training_sequence1, seq1, seq2, l1, l2),
                    '2': (training_sequence2, seq3, seq4, l3, l4),
                    '3': (training_sequence3, seq5, seq6, l5, l6),
                    '4': (training_sequence4, seq7, seq8, l7, l8)
                   }

stepsize = 5
rounds = np.arange(0, 300, stepsize)
prediction1, prediction2 = [], []
training_opt = '1'

av_prediction_last, p1_prediction_last, p2_prediction_last = [], [], []
prediction1, prediction2, prediction_rand = [], [], []
error1, error2, error_rand = [], [], []
training_error = []

training_seq, test_seq1, test_seq2, label1, label2 = training_options[training_opt]
test_seqs = np.array([test_seq1, test_seq2])

for round in rounds:
    network.train(training_seq, stepsize, 1)

    rand_sequence = [lexicon[i] for i in np.random.randint(0, 10, size=12)]

    training_error.append(network.compute_error_batch(training_seq))

    error1.append(network.compute_error(test_seq1))
    error2.append(network.compute_error(test_seq2))
    error_rand.append(network.compute_error(rand_sequence))
    prediction1.append(network.compute_prediction_last_error(test_seq1))
    prediction2.append(network.compute_prediction_last_error(test_seq2))
    prediction_rand.append(network.compute_prediction_last_error(rand_sequence))

print "cross entropy sequence 1:", error1[-1]
print "cross entropy sequence 2:", error2[-1]

print "prediction error sequence 1:", prediction1[-1]
print "prediction error sequence 2:", prediction2[-1]

# plt.plot(rounds, error1, label=label1+" cross-entropy")
# plt.plot(rounds, error2, label=label2+" cross-entropy")
# plt.plot(rounds, error_rand, label="random sequence, cross_entropy")
# plt.plot(rounds, prediction1, label=label1+ " prediction error")
# plt.plot(rounds, prediction2, label=label2+ " prediction error")
# plt.plot(rounds, prediction_rand, label="random sequence, prediction error")
plt.plot(rounds, training_error)
plt.legend(loc=2)
plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("cross-entropy/prediction error")
plt.show()
