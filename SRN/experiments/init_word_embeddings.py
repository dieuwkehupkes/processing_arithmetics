import sys
import os
import numpy as np
import theano
import pickle
import random
import matplotlib.pylab as plt

# add parent folder to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from SRN_Theano import SRN

# set random seed
np.random.seed(10)
random.seed(10)

# parameters network
network_input_size = 10     # dimensionality of embeddings
network_hidden_size = 10
sigma_init = 0.5
sigma_init_embeddings = 0.2
training_size = 100

# parameters training
batchsize = 20
N_training = 5000
stepsize = 100
rounds = np.arange(0, N_training, stepsize)

# load datasets
L2 = np.array(pickle.load(open('../arithmetic_language/L2.pickle', 'rb')))
embeddings = pickle.load(open('../arithmetic_language/one-hot_-19-19.pickle', 'rb'))

# generate training and test set
division_indices = np.random.permutation(len(L2))
train_sample = L2[division_indices[:training_size]]
training = np.array([[embeddings[word] for word in expression] for expression in train_sample])

test_sample = L2[division_indices[training_size:]]
test = np.array([[embeddings[word] for word in expression] for expression in test_sample])

# generate initial word embeddings
embeddings_size = len(training[0][0])
embeddings_matrix_init = np.random.normal(0, sigma_init_embeddings, (embeddings_size, network_input_size)).astype(theano.config.floatX)

# generate network
network = SRN(network_input_size, network_hidden_size, sigma_init, embeddings=embeddings_matrix_init)  # generate network
network.generate_network_dynamics(word_embeddings=True)
network.test_single_sequence()
network.generate_update_function()

pred_error = []
sse = []

sequence = np.array([training[0]])

len_test = float(len(training))
print "start training"
for round in rounds:
    print round
    # compute prediction error and cross entropy testset
    pred_error_round = 0
    sse_round = 0
    for item in training:
        pred_error_round += network.compute_prediction_error(item) 
        sse_round += network.compute_error(item) 
        print "network prediction:", network.network_prediction(item), "true prediction:", network.true_prediction(item)

    pred = pred_error_round/len_test
    cross = sse_round/len_test

    print "prediction error:", pred
    print "sum squared error:", cross
    print "\n"

    pred_error.append(pred)
    sse.append(cross)

    # shuffle training sequence
    # TODO take this out when batch training is implemented
    np.random.shuffle(training)

    # train all batches for stepsize steps
    network.train(training, stepsize, batchsize)

print "average prediction error last round:", pred_error[-1]
print "sum squared error last round:", sse[-1]

plt.plot(rounds, pred_error, label='prediction error')
plt.plot(rounds, sse, label='sum squared error')
plt.legend()
plt.xlabel("number of training rounds")
plt.ylabel("prediction error/sum squared error")
plt.show()

