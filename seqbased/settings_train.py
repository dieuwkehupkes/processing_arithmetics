# imports
from keras.layers import SimpleRNN, GRU, LSTM
from auxiliary_functions import max_length
from architectures import A1, A4, Seq2Seq
from arithmetics import training_treebank, test_treebank, heldout_treebank
import numpy as np

# numpy seed
seed = 7
save_every = 500                 

# network details
architecture        = A1            # Trainings architecture
recurrent_layer     = GRU   # options: SimpleRNN, GRU, LSTM
size_hidden         = 15            # size of the hidden layer
classifiers         = None

# INPUT
train_embeddings  = True          # set to true for cotraining of embeddings
train_classifier  = True           # set to true for cotraining of classifier layer
train_recurrent   = True          # set to true for cotraining recurrent layer
encoding            = 'random'        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality

n = {GRU: 'GRU', SimpleRNN: 'SRN'}
filename = 'models_run2/'+n[recurrent_layer]+str(size_hidden)+ '_seed' + str(seed) + '_'

# PRETRAIN
# Use this to train an already trained model
pretrained_model = None
# pretrained_model = 'models/GRU_A1_5'
# copy_weights     = ['embeddings', 'recurrent']
copy_weights = None

# TRAINING
nb_epoch            = 1500            # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
optimizer           = 'adam'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
dropout_recurrent   = 0.00           # fraction of the inputs to drop for recurrent gates
sample_weights      = None
# provide a dictionary that maps languages to number of sentences to
# generate for that language.
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
digits                      = np.arange(-10, 11)
languages_train             = training_treebank(seed=seed)
format                      = 'infix'
languages_val              = heldout_treebank(seed=seed)
languages_test              = [(name, treebank) for name, treebank in test_treebank(seed=seed)]
test_separately             = True
maxlen                      = max_length(15)

# VISUALISATION AND LOGS
weights_animation = False           # create an animation of layer weights,
                                    # provide tuple of layer and param_id
verbose = 2                         # verbosity mode

