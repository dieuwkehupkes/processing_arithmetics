from keras.models import Sequential
from keras.layers import SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import theano

# np.random.seed(1)

embeddings = np.random.uniform(-0.5, 0.5, (10, 8)).astype(theano.config.floatX)
network = SRN(8, 8, 0.2, embeddings=embeddings)
network.generate_network_dynamics(word_embeddings=True, classifier=True)

# test and training sequences
lexicon = np.zeros((10,10), float)
np.fill_diagonal(lexicon, 1)
# TODO replace with np.identity
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
batchsize = 2
train_opt = '4'
rounds = np.arange(0, 500, stepsize)
error, error1, error2 = [], [], []
prediction1, prediction2, prediction_rand = [], [], []

training_seq, test_seq1, test_seq2, label1, label2 = training_options[train_opt]

test_seqs = np.array([test_seq1, test_seq2])

for round in rounds:

    rand_sequence = [lexicon[k] for k in np.random.randint(0, 10, size=12)]

    err1, err2 = network.compute_error(np.array(test_seqs))
    error1.append(err1)
    error2.append(err2)
    error.append((err1 + err2)/2)

    # print "error, batch", err
    # raw_input()
    pred_e1, pred_e2 = network.prediction_error_diff(np.array(test_seqs))
    prediction1.append(pred_e1)
    prediction2.append(pred_e2)
    
    pred1, pred2 = network.predictions(np.array(test_seqs))
    true_pred1, true_pred2 = network.target_predictions(np.array(test_seqs))
    # print "\nprediction sequence 1", pred1, "\ttrue prediction sequence 1", true_pred1
    # print "prediction sequence 2", pred2, "\ttrue prediction sequence 2", true_pred2
    # raw_input()
    # print "prediction error 1", pred1
    # print "prediction error 2", pred2
    # raw_input()
    err = network.compute_error(np.array([test_seq1, test_seq2]))

    # pred1 = network.prediction_error_diff(np.array([test_seq1]))
    # pred2 = network.prediction_error_diff(np.array([test_seq2]))
    # prediction1.append(pred1)
    # prediction2.append(pred2)
    pred = network.prediction_error_diff(np.array([test_seq1, test_seq2]))

    """
    # print "\ninput map: ", network.print_input_map(test_seqs)
    print "network predictions: ", network.print_predictions(test_seqs)
    print "target predictions: ", network.print_target_predictions(test_seqs)
    raw_input()
    """

    network.train(training_seq, stepsize, batchsize)

print "\nSSE:", error[-1]

print "prediction rate sequence 1:", prediction1[-1]
print "prediction rate sequence 2:", prediction2[-1]

plt.plot(rounds, error1, label=label1+" cross entropy")
plt.plot(rounds, error2, label=label2+" cross entropy")
# plt.plot(rounds, prediction1, label=label1 + " prediction error")
# plt.plot(rounds, prediction2, label=label2 + " prediction error")
plt.legend(loc=2)
# plt.ylim(ymin=0)
plt.xlabel("number of training rounds")
plt.ylabel("sum squared error/prediction error")
plt.show()
