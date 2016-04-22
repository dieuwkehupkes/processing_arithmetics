# imports
from keras.models import Graph
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM
from TrainingHistory import TrainingHistory
from auxiliary_functions import generate_embeddings_matrix
from generate_training_data import generate_training_data
import matplotlib.pyplot as plt

# network details
architecture        = 'A1'          # Trainings architecture (change this)
recurrent_layer     = SimpleRNN     # options: SimpleRNN, GRU, LSTM
size_hidden         = 20            # size of the hidden layer
size_compare        = 10            # size of comparison layer
output_activation   = 'linear'      # activation function of output layer

# INPUT
embeddings          = True          # set to true for cotraining of embeddings
encoding            = 'random'      # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality
# Compute input dimension and input length from training data

# TRAINING
nb_epoch            = 5000          # number of iterations
batch_size          = 20           # batchsize during training
validation_split    = 0.1           # fraction of data to use for testing
verbose             = 1             # verbosity mode
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
# provide a dictionary that maps languages to number of sentences to
# generate for that language. 
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
languages           = {'L_2':1000}            # dict L -> N

# GENERATE TRAINING DATA
X_train, Y_train, N_digits, N_operators = generate_training_data(languages, architecture)

# Generate embeddings matrix
# TODO change this such that it also includes brackets and operators
W_embeddings = generate_embeddings_matrix(N_digits, N_operators, input_size, encoding)
input_dim = N_operators + N_digits + 3      # TODO WHY?????
print input_dim

# TODO write function to compute this
input_length = len(X_train[0])

# Create model
model = Graph()
model.add_input(name='input', input_shape=(1,), dtype='int')        # input layer
model.add_node(Embedding(input_dim=input_dim, output_dim=input_size, input_length=input_length, weights=W_embeddings, mask_zero=mask_zero, trainable=embeddings), name='embeddings', input='input')      # embeddings layer
model.add_node(recurrent_layer(size_hidden), name='recurrent_layer', input='embeddings')    # recurrent layer
model.add_node(Dense(size_compare), name='comparison', input='recurrent_layer')         # comparison layer
model.add_node(Dense(1, activation=output_activation), name='output', input='comparison', create_output=True)

# compile model
model.compile(loss={'output':'mean_squared_error'}, metrics=['accuracy'], optimizer=optimizer)
model.summary()

# train the model
history = TrainingHistory()
model.fit({'input':X_train, 'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history], validation_split=validation_split, shuffle=True)

# plot results
plt.plot(history.losses, label='loss')
plt.title("put title")
plt.xlabel("put x label")
plt.ylabel("put y label")
plt.show()
