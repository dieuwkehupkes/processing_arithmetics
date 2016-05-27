import argparse
from generate_training_data import generate_dmap
from auxiliary_functions import generate_embeddings_matrix, print_sum
from architectures import A1, A4
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running")
# add arguments to override settings file?

args = parser.parse_args()

# import parameters
import_string = args.settings
py = re.compile('\.py$')
if py.search(import_string):
    # take of .py
    import_string = import_string[:-3]

settings = __import__(import_string)

# print model summary
print_sum(settings)

# GENERATE map from words to vectors
dmap, N_operators, N_digits = generate_dmap(settings.digits, settings.languages_train,
                                            settings.languages_val, settings.languages_test)
if settings.pretrained_model:
    print("\nWarning: dmaps should be equal!\n")
    # TODO you should also build in some dimensionality check in here somehow,
    # to check if the test items have correct dimensionaliy

# CREATE TRAININGS ARCHITECTURE
training = settings.architecture()

# GENERATE TRAINING DATA
X, Y = training.generate_training_data(settings.languages_train, dmap=dmap,
                              digits=settings.digits, pad_to=settings.maxlen)

# TODO
if settings.architecture == A1:
    print Y
    print "hier lijkt echt iets raars te gebeuren met A1 training, waarom zijn de targets strings?"

# GENERATE VALIDATION DATA
if settings.languages_val:
    # generate validation data if dictionary is provided
    X_val, Y_val = training.generate_training_data(settings.languages_val, dmap=dmap,
                                                   digits=settings.digits, pad_to=settings.maxlen)
    validation_data = X_val, Y_val
    validation_split = 0.0

else:
    validation_data = None
    validation_split = settings.validation_split

# COMPUTE NETWORK DIMENSIONS
input_dim = len(dmap)+1
input_length = settings.maxlen

# GENERATE EMBEDDINGS MATRIX
W_embeddings = generate_embeddings_matrix(N_digits, N_operators, settings.input_size, settings.encoding)

if settings.pretrained_model:
    model_string = settings.pretrained_model+ '.json'
    model_weights = settings.pretrained_model + '_weights.h5'
    training.add_pretrained_model(model_string, model_weights, optimizer=settings.optimizer, dmap=dmap)
else:
    training.generate_model(settings.recurrent_layer, input_dim=input_dim, input_size=settings.input_size,
                            input_length=input_length, size_hidden=settings.size_hidden,
                            size_compare=settings.size_compare, W_embeddings=W_embeddings, dmap=dmap,
                            trainable_comparison=settings.cotrain_comparison, mask_zero=settings.mask_zero,
                            optimizer=settings.optimizer, dropout_recurrent=settings.dropout_recurrent)

training.train(training_data=(X, Y), validation_data=validation_data, validation_split=validation_split,
               batch_size=settings.batch_size, epochs=settings.nb_epoch, verbosity=settings.verbose,
               weights_animation=settings.weights_animation, plot_embeddings=settings.plot_embeddings,
               logger=settings.print_every)

# plot results
if settings.plot_loss:
    training.plot_loss()
if settings.plot_prediction:
    training.plot_metrics_training()
if settings.plot_embeddings == True:
    training.plot_embeddings()
if settings.plot_esp:
    training.plot_esp()

print_sum(settings)

if settings.languages_test:
    # generate test data
    test_data = training.generate_test_data(settings.languages_test, dmap=dmap,
                                   digits=settings.digits, pad_to=settings.maxlen)
    for name, X, Y in test_data:
        acc = training.model.evaluate(X, Y)
        print "Accuracy for for test set %s:" % name,
        print '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(len(acc))])

# save model
if settings.save_model:
    save = raw_input("\nSave model? y/n ")
    if save == 'n' or save == 'N':
        pass
    elif save == 'y' or save == 'Y':
        model_string = raw_input("Provide filename (without extension) ")
        model_json = training.model.to_json()
        open(model_string + '.json', 'w').write(model_json )
        training.model.save_weights(model_string + '_weights.h5')
        pickle.dump(dmap, open(model_string + '.dmap', 'w'))
        hist = training.trainings_history
        trainings_history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
        pickle.dump(trainings_history, open(model_string + '.history', 'w'))
