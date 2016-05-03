# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1

# network details
architecture        = A1            # Trainings architecture
recurrent_layer     = GRU   # options: SimpleRNN, GRU, LSTM
size_hidden         = 20            # size of the hidden layer
size_compare        = 10            # size of comparison layer

# INPUT
cotrain_embeddings  = True          # set to true for cotraining of embeddings
cotrain_comparison  = True          # set to true for cotraining of comparison layer
encoding            = 'random'        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality

# TRAINING
nb_epoch            = 2000          # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
verbose             = 1             # verbosity mode
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
# provide a dictionary that maps languages to number of sentences to
# generate for that language.
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
# languages           = {'L_2':5, 'L_3':5}            # dict L -> N
languages_train             = {'L_3left': 2000}                 # dict L -> N
languages_val               = None
maxlen                      = max_length(3)

# VISUALISATION
embeddings_animation = False
plot_loss = False
plot_prediction = True
plot_embeddings = 100

# SAVE MODEL
save_model = True
