import argparse
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix
from architectures import A1

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running")
# add arguments to override settings file?

args = parser.parse_args()

# import parameters
settings = __import__(args.settings)

# GENERATE TRAINING DATA
X, Y, N_digits, N_operators, d_map = generate_training_data(settings.languages_train, architecture='A1', pad_to=settings.maxlen)

# GENERATE VALIDATION DATA
if settings.languages_val:
    # generate validation data if dictionary is provided
    X_train, Y_train = X, Y
    X_val, Y_val, _, _, _ = generate_training_data(settings.languages_val, architecture='A1', pad_to=settings.maxlen)

else:
    # split data in training and validation data
    # TODO I suppose this doesn't work for architecture 3 and 4, adapt this later
    split_at = int(len(X)* (1. - settings.validation_split))
    X_train, X_val = X[:split_at], X[split_at:]
    Y_train, Y_val = Y[:split_at], Y[split_at:]

# COMPUTE NETWORK DIMENSIONS
input_dim = N_operators + N_digits + 2
input_length = len(X_train[0])

# GENERATE EMBEDDINGS MATRIX
W_embeddings = generate_embeddings_matrix(N_digits, N_operators, settings.input_size, settings.encoding)

# CREATE TRAININGS ARCHITECTURE
training = A1(settings.recurrent_layer, input_dim=input_dim, input_size=settings.input_size,
              input_length=input_length, size_hidden=settings.size_hidden,
              size_compare=settings.size_compare, W_embeddings=W_embeddings, dmap=d_map,
              trainable_comparison=settings.cotrain_comparison, mask_zero=settings.mask_zero,
              optimizer=settings.optimizer)

training.train(training_data=(X_train, Y_train), validation_data=(X_val, Y_val),
               batch_size=settings.batch_size, epochs=settings.nb_epoch, verbosity=settings.verbose,
               embeddings_animation=settings.embeddings_animation, plot_embeddings=settings.plot_embeddings,
               logger=settings.print_every)

# plot results
if settings.plot_loss:
    training.plot_loss()
if settings.plot_prediction:
    training.plot_prediction_error()
if settings.plot_embeddings == True:
    training.plot_embeddings()

# save model
if settings.save_model:
    # generate name for model
    model_string = ''
    for language in settings.languages_train.keys():
        model_string += language

    model_string += "-"+str(settings.recurrent_layer)

    model_json = training.model.to_json()
    open(model_string + '.json', 'w').write(model_json )
    training.model.save_weights(model_string + '_weights.h5')

print('Model summary:')
print('Recurrent layer: %s' % str(settings.recurrent))
print('Size hidden layer: %i' % settings.size_hidden)
print('Size comparison layer: %i' % settings.size_compare)
print('Initialisation embeddings: %s' % settings.encoding)
print('Size embeddings: %i' % settings.input_size)
print('Batch size: %i' % settings.batch_size)
print('Number of epochs: %i' % settings.nb_epoch)
print('Optimizer: %s' % settings.optimizer)
print('Trained on:')
for language, nr in settings.languages_train.items():
    print('%i sentences from %s' % (nr, language))
