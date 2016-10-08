# imports
from keras.layers import SimpleRNN, GRU, LSTM
import sys
sys.path.insert(0, '../commonFiles')
from arithmetics import test_treebank
from auxiliary_functions import max_length
from architectures import A1, A4, Probing, Seq2Seq
from collections import OrderedDict
import numpy as np


architecture = Seq2Seq
models = 'models_probe/GRU15_seed0probe_seed10_500.h5', 'models_probe/GRU15_seed0probe_seed20_500.h5', 'models_probe/GRU15_seed1probe_seed10_500.h5', 'models_probe/GRU15_seed1probe_seed20_500.h5'
# models = 'models_run2/GRU15postfix_seed0_1500.h5', 'models_run2/GRU15postfix_seed1_1500.h5',  'models_run2/GRU15postfix_seed2_1500.h5', 'models_run2/GRU15postfix_seed3_1500.h5'
models = 'best_models/GRU15_seed1_1500.h5', 'A1/GRU15_seed0_1500.h5'
models = 'models/GRU_A1_1.h5', 'models/GRU_A1_2.h5'
dmap = 'models/dmap'
classifiers = ['intermediate_recursively', 'intermediate_locally', 'subtracting']
classifiers = None
seed = 100
format = 'infix'

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mean_squared_prediction_error', 'binary_accuracy', 'mean_squared_error']         # metrics to be monitored


# optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
# loss = 'categorical_crossentropy'            # loss function
# metrics = ['categorical_accuracy']         # metrics to be monitored

digits = np.arange(-10, 11)

# TEST SETS
# test_sets = ['test_sets/L3_500.test']
# create big test_set

L = [('L1',500), ('L2',500), ('L3',500), ('L4',1500), ('L5',1500), ('L6',2500), ('L7', 3500), ('L8',4500), ('L9', 5500), ('L10', 6500)]
L_left = [('L1_left',500), ('L2_left',500), ('L3_left',500), ('L4_left',1500), ('L5_left',1500), ('L6_left',2500), ('L7_left', 3500), ('L8_left',4500), ('L9_left', 5500), ('L10_left_left', 6500)]
L_right = [('L1_right',500), ('L2_right',500), ('L3_right',500), ('L4_right',1500), ('L5_right',1500), ('L6_right',2500), ('L7_right', 3500), ('L8_right',4500), ('L9_right', 5500), ('L10_right', 6500)]
L_test = L + L_left + L_right

languages_test = OrderedDict([('L1', 50), ('L2', 500), ('L3', 1500), ('L4', 3000), ('L5', 5000), ('L6', 10000), ('L7', 15000), ('L8', 15000), ('L9', 15000), ('L9_left', 15000)])

test_sets = [(name, treebank) for name, treebank in test_treebank(seed=seed)]

test_separately = True
crop_to_length = True


