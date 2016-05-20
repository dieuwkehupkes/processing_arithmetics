# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4
import numpy as np


# NETWORK FILES
architecture = A1
model_architecture = 'model_test.json'         # name of file containing model architecture
model_weights = 'model_test_weights.h5'     # name of file containing model weights
model_dmap = 'model_test.dmap'              # dmap of the embeddings layer of the model

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mspe']         # metrics to be monitored
digits = np.arange(-19, 20)

# TEST SETS
# test_sets = ['test_sets/L3_500.test']
test_sets = {'L3': 500}

# PARAMETERS FOR RUNNING
compute_accuracy = False     # compute accuracy on testset

compute_correls = False      # compute correlation between hidden unit activations

# - overall accuracy op testset berekenen voor alle metrics
# - correlatie tussen de hidden units?
# - run one by one and plot hidden layer activations (animatie/plot) (bij voorkeur met de huidige input op x-as
# - run one by one and print test items and outcomes?
visualise_test_items = True

