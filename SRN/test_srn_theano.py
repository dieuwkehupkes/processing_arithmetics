from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt

network = SRN(10, 8, 0.2)
network.generate_update_function()
network.generate_network_dynamics()


# test and training sequences
lexicon = np.zeros((10,10), float)
np.fill_diagonal(lexicon, 1)
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

stepsize = 1
rounds = np.arange(0, 800, stepsize)
prediction1, prediction2 = [], []

for training_opt in ['4']: #['1', '2', '3', '4']:

    av_prediction_last, p1_prediction_last, p2_prediction_last = [], [], []
    prediction1, prediction2, prediction_rand = [], [], []

    training_seq, test_seq1, test_seq2, label1, label2 = training_options[training_opt]

    for round in rounds:
        network.train(training_seq, stepsize, 1)

        rand_sequence = [lexicon[i] for i in np.random.randint(0, 10, size=12)]

        p1, p2 = network.predict_last(test_seq1), network.predict_last(test_seq2)

        p1_prediction_last.append(p1)
        p2_prediction_last.append(p2)
        av_prediction_last.append(0.5 * (p1+p2))
        prediction1.append(network.compute_prediction_error(test_seq1))
        prediction2.append(network.compute_prediction_error(test_seq2))

    print "prediction rate sequence 1: ", prediction1[-1]
    print "prediction rate sequence 2: ", prediction2[-1]

    # plt.plot(rounds, prediction1, label=label1+ " prediction error")
    # plt.plot(rounds, prediction2, label=label2+ " prediction error")
    plt.plot(rounds, p1_prediction_last, label = label1+" prediction last")
    plt.plot(rounds, p2_prediction_last, label = label2+" prediction last")

plt.legend(loc=2)
plt.ylim(ymin=0, ymax=1.5)
plt.xlabel("number of training rounds")
plt.ylabel("prediction error")
plt.show()
# 
# for training_opt in ['4']: #training_options:
#     training_seq, test_seq1, test_seq2, label1, label2 = training_options[training_opt]
#     e1, e2 = [], []
#     for round in rounds:
#         network.train(training_seq, stepsize)
#         e1.append(network.compute_prediction_error(test_seq1))
#         e2.append(network.compute_prediction_error(test_seq2))
# 
#     plt.plot(rounds, e1, label=label1)
#     plt.plot(rounds, e2, label=label2)
# 
# plt.legend(loc=2)
# plt.ylim(ymin=0, ymax=1)
# plt.xlabel("number of training rounds")
# plt.ylabel("prediction error")
# 
# plt.show()

exit()

training_seq, test_seq1, test_seq2, label1, label2 = training_options[train_opt]

for round in rounds:
    network.train(training_seq, stepsize)

    rand_sequence = [lexicon[i] for i in np.random.randint(0, 10, size=12)]

    error1.append(network.compute_error(test_seq1))
    error2.append(network.compute_error(test_seq2))
    error_rand.append(network.compute_error(rand_sequence))
    prediction1.append(network.compute_prediction_error(test_seq1))
    prediction2.append(network.compute_prediction_error(test_seq2))
    prediction_rand.append(network.compute_prediction_error(rand_sequence))

print "cross entropy sequence 1:", error1[-1]
print "cross entropy sequence 2:", error2[-1]

print "prediction rate sequence 1:", prediction1[-1]
print "prediction rate sequence 2:", prediction2[-1]

plt.plot(rounds, error1, label=label1+" cross-entropy")
plt.plot(rounds, error2, label=label2+" cross-entropy")
plt.plot(rounds, error_rand, label="random sequence, cross_entropy")
plt.plot(rounds, prediction1, label=label1+ " prediction error")
plt.plot(rounds, prediction2, label=label2+ " prediction error")
plt.plot(rounds, prediction_rand, label="random sequence, prediction error")
plt.legend(loc=2)
plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("cross-entropy/prediction error")
plt.show()
