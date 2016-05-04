# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1

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

# TRAINING
nb_epoch            = 2500          # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
# provide a dictionary that maps languages to number of sentences to
# generate for that language.
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
# languages           = {'L_2':5, 'L_3':5}            # dict L -> N
languages_train             = {'L2+': 1000, 'L3+':2000, 'L4+':3000}                 # dict L -> N
languages_val               = {'L5+':500}
maxlen                      = max_length(5)

# VISUALISATION AND LOGS
embeddings_animation = False        # create an animation of embeddings development
plot_loss = False                   # plot loss
plot_prediction = True              # plot prediction error
plot_embeddings = 500               # create scatterplot of embeddings
verbose = 1                         # verbosity mode
print_every = False                 # print results

# SAVE MODEL
save_model = True
