# Probe what is encoded in the representations of a model
# by retraining different classifiers on its intermediate
# representations and testing how accurately they can extract
# different features from these representation

import argparse
import re
import pickle
from auxiliary_functions import generate_embeddings_matrix, print_sum
from architectures import Probing, A1


def probe_model(architecture, weights, dmap, classifiers, digits, languages_train, optimizer, dropout_recurrent, batch_size, nb_epochs, validation_split, verbosity, maxlen):

    training2 = Probing(classifiers)

    training1 = A1()

    print "model 1"

    training1.add_pretrained_model(architecture, weights, dmap, copy_weights=['recurrent', 'embeddings'], train_classifier=True, train_embeddings=False, train_recurrent=False, mask_zero=True, dropout_recurrent=dropout_recurrent, optimizer=optimizer)

    # print training1.model.summary()

    print "\nmodel 2"

    training2.add_pretrained_model(architecture, weights, dmap, copy_weights=['recurrent', 'embeddings'], train_classifier=True, train_embeddings=False, train_recurrent=False, mask_zero=True, dropout_recurrent=dropout_recurrent, optimizer=optimizer)

    # print training2.model.summary()
 
    dmap = pickle.load(open(dmap, 'rb'))

    print "training data A1: "
    x_train, y_train = A1.generate_training_data(languages_train, dmap, digits, pad_to=maxlen)

    print x_train.shape
    print "output shape: ", y_train.shape
    print y_train

    # training1.train((x_train, y_train), batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split, verbosity=verbosity)

    print "training data Probe: "

    X_train, Y_train = training2.generate_training_data(languages_train, dmap, digits, classifiers, pad_to=maxlen)

    print X_train.shape
    print "output shape: ", Y_train['grammatical'].shape

    training2.train((X_train, Y_train), batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split, verbosity=verbosity)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="Provide file with settings for model running", default="settings_train")

    args = parser.parse_args()

    # import parameters
    import_string = args.settings
    py = re.compile('\.py$')
    if py.search(import_string):
        # take of .py
        import_string = import_string[:-3]

    settings = __import__(import_string)

    probe_model(architecture=settings.model_architecture, weights=settings.model_weights, dmap=settings.dmap, digits=settings.digits, languages_train=settings.languages_train, classifiers=settings.classifiers, optimizer=settings.optimizer, dropout_recurrent=settings.dropout_recurrent, batch_size=settings.batch_size, maxlen=settings.maxlen, nb_epochs=settings.nb_epochs, validation_split=settings.validation_split, verbosity=settings.verbosity)


