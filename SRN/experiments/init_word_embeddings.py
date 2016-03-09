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
import analyser as an

# set random seed
np.random.seed(2)

# parameters network
network_input_size = 10     # dimensionality of embeddings
network_hidden_size = 10
sigma_init = 0.5
sigma_init_embeddings = 0.2
training_size = 500
learning_rate = 0.5

# parameters training
batchsize = 250
N_training = 50000
stepsize = 500
rounds = np.arange(0, N_training, stepsize)

# load datasets
L2 = np.array(pickle.load(open('arithmetic_language/L2.pickle', 'rb')))
# L2 = np.array(pickle.load(open('../arithmetic_language/L2.pickle', 'rb')))
embeddings = pickle.load(open('arithmetic_language/one-hot_-19-19.pickle', 'rb'))
# embeddings = pickle.load(open('../arithmetic_language/one-hot_-19-19.pickle', 'rb'))

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
network = SRN(network_input_size, network_hidden_size, sigma_init, embeddings=embeddings_matrix_init, learning_rate=learning_rate)  # generate network
network.generate_network_dynamics(word_embeddings=True, classifier=True)

pred_error = []
sse = []

sequence = np.array([training[0]])

len_test = float(len(training))
print "start training"
for round in rounds:
    # compute training error
    pred = network.mean_squared_prediction_error(training)
    cross = np.mean(network.compute_error(training))
 
    # print "\naverage distance between word embeddings:", an.distance_embeddings(network.embeddings.get_value())
    # print "average length word embeddings:", an.length_embeddings(network.embeddings.get_value())
    # an.plot_distances(network.embeddings.get_value())

    print round, "\tprediction error: ", pred, "\tcross entropy: ", cross
    # print "\nprediction error:", pred
    # print "sum squared error:", cross
    # print "\n"

    pred_error.append(pred)
    sse.append(cross)

    # train all batches for stepsize steps
    network.train(training, stepsize, batchsize)

print "average prediction error last round:", pred_error[-1]
print "cross entropy error last round:", sse[-1]

# plot prediction error
plt.plot(rounds, pred_error, label='prediction error')
plt.legend()
plt.xlabel("number of training rounds")
plt.ylabel("prediction error")
plt.ylim(ymin=0)
plt.show()

# plot sum squared error
plt.plot(rounds, sse, label='cross entropy')
plt.xlabel("number of training rounds")
plt.ylabel("cross entropy")
plt.ylim(ymin=0)
plt.show()

