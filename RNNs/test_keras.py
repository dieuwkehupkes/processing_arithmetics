from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, Dense
from keras.utils.visualize_util import plot
from auxiliary_functions import pad
import numpy as np
import matplotlib.pyplot as plt
import theano

# GENERATE TEST/TRAINING DATA

# Lexicon
a, b, c, d, e, f, g, h, i, j = np.identity(10, float)

# generate sequences, padd them to same length
seq1, l1 = pad(np.array([a,b,d]), 6), 'abd'
seq2, l2 = pad(np.array([f,b,i]), 6), 'fbi'
seq3, l3 = pad(np.array([a, b, b, d]), 6), 'abbd'
seq4, l4 = pad(np.array([f, b, b, i]), 6), 'fbbi'
seq5, l5 = pad(np.array([a, b, b, b, d]), 6), 'abbbd'
seq6, l6 = pad(np.array([f, b, b, b, i]), 6), 'fbbbi'
seq7, l7 = pad(np.array([a, b, b, b, b, d]), 6), 'abbbbd'
seq8, l8 = pad(np.array([f, b, b, b, b, i]), 6), 'fbbbbi'

embeddings = np.random.uniform(-0.5, 0.5, (10, 8)).astype(theano.config.floatX)

training_sequence1, targets1 = np.array([seq1[:-1], seq2[:-1]]), np.array([seq1[-1], seq2[-1]])
training_sequence2, targets2 = np.array([seq3[:-1], seq4[:-1]]), np.array([seq3[-1], seq4[-1]])
training_sequence3, targets3 = np.array([seq5[:-1], seq6[:-1]]), np.array([seq5[-1], seq6[-1]])
training_sequence4, targets4 = np.array([seq7[:-1], seq8[:-1]]), np.array([seq7[-1], seq8[-1]])

training_options = {'1': (training_sequence1, seq1, seq2, l1, l2),
                    '2': (training_sequence2, seq3, seq4, l3, l4),
                    '3': (training_sequence3, seq5, seq6, l5, l6),
                    '4': (training_sequence4, seq7, seq8, l7, l8)
                    }


# GENERATE NETWORK

model = Sequential()
# create layer that maps integer between 0-10 to vector of length 8
# generates outputs of shape (batch_dim, 5, 8)
# model.add(Dense(8, input_dim=10))

# Add RNN, output dimensions
model.add(SimpleRNN(output_dim=8, input_length=6, input_dim=10))
model.add(Dense(10, activation='softmax'))

# compile model
print('compile model')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('training sequences')
print training_sequence4
print targets4

exit()

# train model
print('train model')
model.fit(x=training_sequence4, y=targets4, batch_size=2, nb_epoch=20, verbose=2, shuffle=True)


"""
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

    # print "\ninput map: ", network.print_input_map(test_seqs)
    print "network predictions: ", network.print_predictions(test_seqs)
    print "target predictions: ", network.print_target_predictions(test_seqs)
    raw_input()

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
"""
