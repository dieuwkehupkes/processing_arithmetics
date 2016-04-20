# imports
from keras.models import Graph
from keras.layers import SimpleRnn, Embedding, Dense, GRU, LSTM
from TrainingHistory import TrainingHistory
from auxiliary_functions import generate_embeddings_matrix
import matplotlib.pyplot as plt

# network details
recurrent_layer     = GRU           # options: SimpleRNN, GRU, LSTM
size_hidden         = 10            # size of the hidden layer
size_compare        = 10            # size of comparison layer
output_activation   = 'linear'      # activation function of output layer

# Input
embeddings          = True          # set to true for cotraining of embeddings
encoding            = 'random'      # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 2             # input dimensionality
# Compute input dimension and input length from training data
# TODO compute this instead of inputting it
input_length        = 10
input_dim           = 39

# Training
nb_epoch            = 5000          # number of iterations
batch_size          = 100           # batchsize during training
validation_split    = 0.1           # fraction of data to use for testing
verbose             = 0             # verbosity mode
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax

# training data and test data
# TODO add function to generate this?
X_train             = None          # inputs, numpy array
Y_train             = None          # target, numpy array
validation_split    = 0.1           # fraction of data to use for testing


# If required, generate embeddings parameters
W_embeddings = generate_embeddings_matrix(input_dim, input_size, encoding)


# Create model
model = Graph()
model.add_input(name='input', input_shape=(1,), dtype='int')        # input layer
model.add_node(Embedding(input_dim=input_dim, output_dim=input_size, input_length=input_length, weights=W_embeddings, mask_zero=mask_zero, trainable=embeddings), name='embeddings', input='input')      # embeddings layer
model.add_node(recurrent_layer(size_hidden), name='recurrent_layer', input='embeddings')    # recurrent layer
model.add_node(Dense(size_compare), name='comparison', input='recurrent_layer')         # comparison layer
model.add_node(Dense(1, activation=output_activation), name='output', input='comparison', create_output=True)

# compile model
model.compile(loss={'output':'mean_squared_error'}, metrics=['accuracy'], optimizer=optimizer)

# train the model
history = TrainingHistory()
model.fit({'input':X_train, 'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[history], validation_split=validation_split, shuffle=True)

# plot results
plt.plot(history.losses, label='loss')
plt.title("put title")
plt.xaxis("put x label")
plt.yaxis("put y label")
plt.show()
