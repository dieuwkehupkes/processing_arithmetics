# imports
from keras.layers import SimpleRNN, GRU, LSTM
import sys
sys.path.insert(0, '../arithmetics')
from arithmetics import test_treebank
from auxiliary_functions import max_length
from architectures import A1, A4, Probing, Seq2Seq
from collections import OrderedDict
import numpy as np


format = 'infix'
architecture = Probing
models = 'models_probe/GRU15_seed0probe_seed10_500.h5', 'models_probe/GRU15_seed0probe_seed20_500.h5', 'models_probe/GRU15_seed1probe_seed10_500.h5', 'models_probe/GRU15_seed1probe_seed20_500.h5'
# models = 'models_run2/GRU15postfix_seed0_1500.h5', 'models_run2/GRU15postfix_seed1_1500.h5',  'models_run2/GRU15postfix_seed2_1500.h5', 'models_run2/GRU15postfix_seed3_1500.h5'
models = 'best_models/GRU15_seed0probe_seed10_500.h5', 'best_models/GRU15_seed1probe_seed10_500.h5'
dmap = 'models/dmap'
classifiers = ['intermediate_recursively', 'intermediate_locally', 'subtracting']
# classifiers = None
seed = 100

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mean_squared_prediction_error', 'binary_accuracy', 'mean_squared_error']         # metrics to be monitored


# optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
# loss = 'categorical_crossentropy'            # loss function
# metrics = ['categorical_accuracy']         # metrics to be monitored

digits = np.arange(-10, 11)

# TEST SETS
test_sets = [(name, treebank) for name, treebank in test_treebank(seed=seed)]

test_separately = True
crop_to_length = True


