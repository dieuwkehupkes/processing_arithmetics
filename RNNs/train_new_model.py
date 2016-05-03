import argparse
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix
from architectures import A1

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running")
# add arguments to override settings file?

args = parser.parse_args()
settings = args.settings

# import settings
from settings import *

# GENERATE TRAINING DATA
X, Y, N_digits, N_operators, d_map = generate_training_data(languages_train, architecture='A1', pad_to=maxlen)

# GENERATE VALIDATION DATA
if languages_val:
    # generate validation data if dictionary is provided
    X_train, Y_train = X, Y
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
if plot_embeddings == True:
    training.plot_embeddings()

# save model
if save_model:
    # generate name for model
    model_string = ''
    for language in languages_train.keys():
        model_string += language

    model_json = training.model.to_json()
    open(model_string + '.json', 'w').write(model_json )
    training.model.save_weights(model_string + '_weights.h5')
