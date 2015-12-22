import sys
import os
import numpy as np
import theano
import pickle
import random

# add parent folder to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from SRN_Theano import SRN

# set random seed
np.random.seed(10)
random.seed(10)

# parameters network
network_input_size = 10     # dimensionality of embeddings
network_hidden_size = 10
sigma_init = 0.2
sigma_init_embeddings = 0.2
training_size = 3000

# parameters training
batchsize = 200
N_training = 1000
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
network.test_single_sequence

pred_error = []
cross_entr = []

len_test = float(len(test))
for round in rounds:

    # shuffle training sequence
    np.random.shuffle(training)

    # train all batches for stepsize steps
    network.train(training, stepsize, batchsize)

    # compute prediction error and cross entropy testset
    prediction_error = 0
    cross_entropy = 0
    for item in test:
       prediction_error += network.compute_prediction_last_error(item) 
       cross_entropy += network.compute_error(item) 

    pred = prediction_error/len_test
    cross = cross_entropy/len_test
    error.append(pred)
    cross_entr.append(cross)

