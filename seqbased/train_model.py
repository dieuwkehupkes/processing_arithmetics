import sys 
sys.path.insert(0, '../commonFiles') 
import argparse
from generate_training_data import generate_dmap
from auxiliary_functions import generate_embeddings_matrix, print_sum, save_model
from architectures import Training
import re

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running", default="settings_train")
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
# TODO rm classifiers from this func
X, Y = training.generate_training_data(settings.languages_train, dmap=dmap,
                                       digits=settings.digits, format=settings.format,
                                       classifiers=settings.classifiers,
                                       pad_to=settings.maxlen)

# GENERATE VALIDATION DATA
if settings.languages_val:
    # generate validation data if dictionary is provided
    X_val, Y_val = training.generate_training_data(settings.languages_val, dmap=dmap,
                                                   digits=settings.digits,
                                                   classifier=settings.classifiers,
                                                   format=settings.format,
                                                   pad_to=settings.maxlen)
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
    model_string = settings.pretrained_model + '.json'
    model_weights = settings.pretrained_model + '_weights.h5'

    training.add_pretrained_model(model_string, model_weights, 
                                  dmap=dmap, copy_weights=settings.copy_weights,
                                  train_classifier=settings.train_classifier,
                                  train_embeddings=settings.train_embeddings,
                                  train_recurrent=settings.train_recurrent,
                                  mask_zero=settings.mask_zero,
                                  optimizer=settings.optimizer,
                                  dropout_recurrent=settings.dropout_recurrent)

else:
    training.generate_model(settings.recurrent_layer, input_dim=input_dim, input_size=settings.input_size,
                            input_length=input_length, size_hidden=settings.size_hidden,
                            dmap=dmap,
                            W_embeddings=W_embeddings,
                            train_classifier=settings.train_classifier, 
                            train_embeddings=settings.train_embeddings,
                            train_recurrent=settings.train_recurrent,
                            mask_zero=settings.mask_zero,
                            optimizer=settings.optimizer, dropout_recurrent=settings.dropout_recurrent,
                            extra_classifiers=settings.classifiers)


training.train(training_data=(X, Y), validation_data=validation_data, validation_split=validation_split,
               batch_size=settings.batch_size, epochs=settings.nb_epoch, verbosity=settings.verbose,
               weights_animation=settings.weights_animation, plot_embeddings=settings.plot_embeddings,
               logger=settings.print_every)

# plot results
if settings.plot_loss:
    training.plot_loss()
if settings.plot_prediction:
    training.plot_metrics_training()
if settings.plot_embeddings is True:
    training.plot_embeddings()
if settings.plot_esp:
    training.plot_esp()

print_sum(settings)

if settings.languages_test:
    # generate test data
    test_data = Training.generate_test_data(
            settings.architecture,
            settings.languages_test,
            dmap=dmap,
            digits=settings.digits,
            classifiers=settings.classifiers,
            format=settings.format,
            test_separately=settings.test_separately,
            pad_to=settings.maxlen)

    hist = training.trainings_history

    # TODO hier gaat iets mis met printen bij probing
    # print "Accuracy for for training set %s:\t" % \
    #       '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_train.items()])
    # print "Accuracy for for validation set %s:\t" % \
    #       '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_val.items()])
    # for name, X, Y in test_data:
    #     acc = training.model.evaluate(X, Y)
    #     print "Accuracy for for test set %s:" % name,
    #     print '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(len(acc))])

# save model
if settings.save_model:
    save_model(training)
