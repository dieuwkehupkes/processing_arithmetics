# imports
from keras.models import Graph
from keras.layers import SimpleRnn, Embedding, Dense, GRU, LSTM
from TrainingHistory import TrainingHistory

# network details
recurrent_layer     = GRU           # options: SimpleRNN, GRU, LSTM
size_hidden         = 10            # size of the hidden layer
size_compare        = 10            # size of comparison layer
output_activation   = 'linear'      # activation function of output layer

# training data and test data
# TODO add function to generate this?
X_train             = None          # inputs, numpy array
Y_train             = None          # target, numpy array
validation_split    = 0.1           # fraction of data to use for testing

# Input
embeddings          = True          # set to true for cotraining of embeddings
initialisation      = random        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
dimensionality      = 2             # input dimensionality

# Compute input dimension and input length from training data
# TODO compute this instead of inputting it
input_length        = 10
input_dim           = 39

# Training
nb_epoch            = 5000          # number of iterations
batch_size          = 100           # batchsize during training
verbose             = 0             # verbosity mode


# If required, generate embeddings parameters
# TODO write this function, should return none if initialisation is random
W_embeddings = generate_embeddings(input_length, input_dim, embeddings, initialisation, dimensionality)


# Create model
model = Graph()
model.add_input(name='input', input_shape=(1,), dtype='int')        # input layer
model.add_node(Embedding(input_dim=input_dim, output_dim=dimensionalit, input_length=input_length, weights=W_embeddings, mask_zero=mask_zero, trainable=embeddings), name='embeddings', input='input')      # embeddings layer
model.add_node(recurrent_layer(size_hidden), name='recurrent_layer', input='embeddings')    # recurrent layer
model.add_node(Dense(size_compare), name='comparison', input='recurrent_layer')         # comparison layer
model.add_node(Dense(1, activation=output_activation), name='output', input='comparison', create_output=True)

# train the model
history = TrainingHistory()
model.fit({'input':X_train, 'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[history], validation_split=validation_split, shuffle=True)

# plot results
plt.plot(history.losses, label='loss')
plt.title("put title")
plt.xaxis("put x label")
plt.yaxis("put y label")
plt.show()
