# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4
import numpy as np

# network details
architecture        = A1            # Trainings architecture
recurrent_layer     = SimpleRNN   # options: SimpleRNN, GRU, LSTM
size_hidden         = 15            # size of the hidden layer
size_compare        = 2            # size of comparison layer

# INPUT
cotrain_embeddings  = True          # set to true for cotraining of embeddings
cotrain_comparison  = True          # set to true for cotraining of comparison layer
encoding            = 'random'        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality

# PRETRAIN
# Use this to train an already trained model
# pretrained_model = 'test_model'
pretrained_model = None

# TRAINING
nb_epoch            = 1000            # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
optimizer           = 'adam'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
dropout_recurrent   = 0.00           # fraction of the inputs to drop for recurrent gates
# provide a dictionary that maps languages to number of sentences to
# generate for that language.
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
digits                      = np.arange(-10, 11)
languages_train             = {'L2': 2000}
languages_val               = None
# languages_val               = {'L4': 500}
# languages_test              = {'L2': 2000, 'L3':500, 'L6':500}
languages_test              = {'L2': 2000}
maxlen                      = max_length(6)

# VISUALISATION AND LOGS
weights_animation = False           # create an animation of layer weights,
                                    # provide tuple of layer and param_id
plot_loss = False                   # plot loss
plot_prediction = True              # plot prediction error
plot_embeddings = 200               # create scatterplot of embeddings
plot_esp = False                     # plot spectral radius of recurrent connections
verbose = 1                         # verbosity mode
print_every = False                 # print results

# SAVE MODEL
save_model = True
