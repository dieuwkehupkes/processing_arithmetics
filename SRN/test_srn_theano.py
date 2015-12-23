from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt
import theano

np.random.seed(1)

embeddings = np.random.normal(0, 0.2, (10, 8)).astype(theano.config.floatX)
network = SRN(8, 8, 0.2, embeddings=embeddings)
# network = SRN(10, 8, 0.2)
network.generate_network_dynamics(word_embeddings=False)
network.test_single_sequence()

# test and training sequences
lexicon = np.zeros((10,10), float)
np.fill_diagonal(lexicon, 1)
# TODO replace with np.identity
a, b, c, d, e, f, g, h, i, j = lexicon

seq1, l1 = np.array([a,b,d]), 'abd'
seq2, l2 = np.array([f,b,i]), 'fbi'
seq3, l3 = np.array([a, b, b, d]), 'abbd'
seq4, l4 = np.array([f, b, b, i]), 'fbbi'
seq5, l5 = np.array([a, b, b, d]), 'abbbd'
seq6, l6 = np.array([f, b, b, i]), 'fbbbi'
seq7, l7 = np.array([a, b, b, d]), 'abbbbd'
seq8, l8 = np.array([f, b, b, i]), 'fbbbbi'


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
batchsize = 2
train_opt = '4'
rounds = np.arange(0, 5000, stepsize)
error1, error2, error_rand = [], [], []
prediction1, prediction2, prediction_rand = [], [], []

training_seq, test_seq1, test_seq2, label1, label2 = training_options[train_opt]

network.train(training_seq, stepsize, batchsize)

for round in rounds:
    network.train(training_seq, stepsize, batchsize)

    rand_sequence = [lexicon[k] for k in np.random.randint(0, 10, size=12)]

    error1.append(network.compute_error(test_seq1))
    error2.append(network.compute_error(test_seq2))
    error_rand.append(network.compute_error(rand_sequence))
    prediction1.append(network.compute_prediction_error(test_seq1))
    prediction2.append(network.compute_prediction_error(test_seq2))
    prediction_rand.append(network.compute_prediction_error(rand_sequence))

print "\ncross entropy sequence 1:", error1[-1]
print "cross entropy sequence 2:", error2[-1]

print "prediction rate sequence 1:", prediction1[-1]
print "prediction rate sequence 2:", prediction2[-1]

plt.plot(rounds, error1, label=label1+" sum squared error")
plt.plot(rounds, error2, label=label2+" sum squared error")
plt.plot(rounds, error_rand, label="random sequence, cross_entropy")
# plt.plot(rounds, prediction1, label=label1 + " prediction error")
# plt.plot(rounds, prediction2, label=label2 + " prediction error")
# plt.plot(rounds, prediction_rand, label="random sequence, prediction error")
plt.legend(loc=2)
# plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("sum squared error/prediction error")
plt.show()
