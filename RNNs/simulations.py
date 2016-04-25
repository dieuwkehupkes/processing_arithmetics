# imports
from keras.layers import SimpleRNN, GRU, LSTM
from architectures import A1, A2, A3, A4
import matplotlib.pyplot as plt

# network details
architecture        = A1            # Trainings architecture
recurrent_layer     = SimpleRNN     # options: SimpleRNN, GRU, LSTM
size_hidden         = 20            # size of the hidden layer
size_compare        = 10            # size of comparison layer

# INPUT
cotrain             = True          # set to true for cotraining of embeddings
encoding            = 'gray'        # options: random, gray
mask_zero           = True          # set to true to apply masking to input
input_size          = 6             # input dimensionality
# Compute input dimension and input length from training data

# TRAINING
nb_epoch            = 1500          # number of iterations
batch_size          = 24            # batchsize during training
validation_split    = 0.1          # fraction of data to use for testing
verbose             = 1             # verbosity mode
optimizer           = 'adagrad'     # sgd, rmsprop, adagrad, adadelta, adam, adamax
# provide a dictionary that maps languages to number of sentences to
# generate for that language. 
# languages \in L_i, L_i+, L_i-, L_iright, L_ileft for 1<i<8)
# languages           = {'L_2':5, 'L_3':5}            # dict L -> N
languages           = {'L_2':1000}                    # dict L -> N

history = architecture(languages=languages, input_size=input_size, size_hidden=size_hidden, size_compare=size_compare, recurrent=recurrent_layer, encoding=encoding, trainable=cotrain, mask_zero=mask_zero, optimizer=optimizer, validation_split=validation_split, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)

# raw_input()

# plot results
plt.plot(history.losses, label='loss training set')
plt.plot(history.val_losses, label='loss validation set')
plt.title("Loss function over epoch")
plt.xlabel("Epoch")
plt.ylabel("Sum squared error")
plt.show()
