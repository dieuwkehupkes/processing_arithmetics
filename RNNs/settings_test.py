# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4
import numpy as np


# NETWORK FILES
architecture = A4
model_architecture = 'test_model_A4.json'         # name of file containing model architecture
model_weights = 'test_model_A4_weights.h5'     # name of file containing model weights
model_dmap = 'test_model_A4.dmap'              # dmap of the embeddings layer of the model

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'categorical_crossentropy'            # loss function
metrics = ['categorical_accuracy']         # metrics to be monitored
digits = np.arange(-10, 11)

# TEST SETS
# test_sets = ['test_sets/L3_500.test']
test_sets = {'L2': 500}


