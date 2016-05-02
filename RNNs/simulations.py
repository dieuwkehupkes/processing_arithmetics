# imports
from keras.layers import SimpleRNN, GRU, LSTM
from architectures import A1
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix

# network details
architecture        = A1            # Trainings architecture
recurrent_layer     = SimpleRNN     # options: SimpleRNN, GRU, LSTM
size_hidden         = 20            # size of the hidden layer
size_compare        = 10            # size of comparison layer

# INPUT
cotrain_embeddings  = True          # set to true for cotraining of embeddings
cotrain_comparison  = True          # set to true for cotraining of comparison layer
encoding            = 'random'        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality

# TRAINING
nb_epoch            = 3500          # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
verbose             = 1             # verbosity mode
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
# provide a dictionary that maps languages to number of sentences to
# generate for that language. 
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
# languages           = {'L_2':5, 'L_3':5}            # dict L -> N
languages_train             = {'L_2+': 2000, 'L4+':2000, 'L6+': 2000}                    # dict L -> N
languages_val               = {'L_4+': 500}
# languages_val               = None

# VISUALISATION
embeddings_animation = False
plot_loss = False
plot_prediction = True
plot_embeddings = 500



#########################################################################################
#########################################################################################


# GENERATE TRAINING DATA
X, Y, N_digits, N_operators, d_map = generate_training_data(languages_train, architecture='A1')

# GENERATE VALIDATION DATA
if languages_val:
    # generate validation data if dictionary is provided
    X_train, Y_train = X, Y
    maxlen = X_train.shape[1]
    X_val, Y_val, _, _, _ = generate_training_data(languages_val, architecture='A1', pad_to=maxlen)

else:
    # split data in training and validation data
    # TODO I suppose this doesn't work for architecture 3 and 4, adapt this later
    split_at = int(len(X)* (1. - validation_split))
    X_train, X_val = X[:split_at], X[split_at:]
    Y_train, Y_val = Y[:split_at], Y[split_at:]

# COMPUTE NETWORK DIMENSIONS
input_dim = N_operators + N_digits + 2
input_length = len(X_train[0])

# GENERATE EMBEDDINGS MATRIX
W_embeddings = generate_embeddings_matrix(N_digits, N_operators, input_size, encoding)

# CREATE TRAININGS ARCHITECTURE
training = A1(recurrent_layer, input_dim=input_dim, input_size=input_size, input_length=input_length,
              size_hidden=size_hidden, size_compare=size_compare, W_embeddings=W_embeddings, dmap=d_map,
              trainable_comparison=cotrain_comparison, mask_zero=mask_zero, optimizer=optimizer)

training.train(training_data=(X_train, Y_train), validation_data=(X_val, Y_val), batch_size=batch_size, epochs=nb_epoch, embeddings_animation=embeddings_animation, plot_embeddings=plot_embeddings)

# plot results
if plot_loss:
    training.plot_loss()
if plot_prediction:
    training.plot_prediction_error()
if plot_embeddings:
    training.plot_embeddings()

