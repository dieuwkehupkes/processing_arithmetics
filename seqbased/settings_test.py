# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4
from collections import OrderedDict
import numpy as np


architecture = A1
prefs = 'models/SRN_A1_1', 'models/SRN_A1_2', 'models/SRN_A1_3', 'models/SRN_A1_4'

# SETTINGS OF NETWORK
optimizer = 'adam'      # sgd, rmsprop, adagrad, adadelta, adam of adamax
loss = 'mse'            # loss function
metrics = ['mean_squared_prediction_error']         # metrics to be monitored


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

L_test = [('L1',5000), ('L2',5000), ('L3',5000), ('L5',10000), ('L6',10000), ('L7', 10000), ('L9', 10000), ('L9_left',15000)]

test_sets = OrderedDict(L_test)
test_separately = True


