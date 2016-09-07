# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4
import numpy as np


# NETWORK FILES
architecture = A1
pref = 'models/GRU_A1_2'
model_architecture = pref+'.json'         # name of file containing model architecture
model_weights = pref+'_weights.h5'     # name of file containing model weights
model_dmap = pref+'.dmap'              # dmap of the embeddings layer of the model

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mean_squared_prediction_error']         # metrics to be monitored

# optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
# loss = 'mse'            # loss function
# metrics = ['mspe']         # metrics to be monitored

digits = np.arange(-10, 11)

# TEST SETS
# test_sets = {'L1': 500, 'L2': 500, 'L3': 500, 'L4': 500, 'L5': 500, 'L6': 500, 'L7': 500}
test_sets = {'L3': 500}

# PARAMETERS FOR RUNNING
compute_accuracy = True     # compute accuracy on testset

compute_correls = False      # compute correlation between hidden unit activations

project_lexical = False      # compute projections of lexical items

visualise_paths = False         # integer, how many paths to visualise

plot_activations = True

# - overall accuracy op testset berekenen voor alle metrics
# - correlatie tussen de hidden units?
# - run one by one and plot hidden layer activations (animatie/plot) (bij voorkeur met de huidige input op x-as
# - run one by one and print test items and outcomes?
one_by_one = 3
plot_gate_values = False

