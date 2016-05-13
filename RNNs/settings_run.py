# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1
import numpy as np


# NETWORK FILES
model_architecture = 'model_test.json'         # name of file containing model architecture
model_weights = 'model_test_weights.h5'     # name of file containing model weights
model_dmap = 'model_test.dmap'              # dmap of the embeddings layer of the model

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mspe']         # metrics to be monitored

# TEST SETS
test_set = 'test_sets/L3_500.test'


# PARAMETERS FOR RUNNING
compute_accuracy = True     # compute accuracy on testset

# - overall accuracy op testset berekenen voor alle metrics
# - correlatie tussen de hidden units?
# - run one by one and plot hidden layer activations (animatie/plot) (bij voorkeur met de huidige input op x-as
# - run one by one and print test items and outcomes?
one_by_one = True

