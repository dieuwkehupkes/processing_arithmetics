# Probe what is encoded in the representations of a model
# by retraining different classifiers on its intermediate
# representations and testing how accurately they can extract
# different features from these representation

import sys 
sys.path.insert(0, '../commonFiles') 
from train_model import generate_training_data, generate_test_data
import argparse
import re
import pickle
from auxiliary_functions import generate_embeddings_matrix, print_sum, save_model
from architectures import Probing, A1, Training


def train_model(model, dmap, classifiers, digits, languages_train, optimizer, dropout_recurrent, batch_size, nb_epochs, validation_split, sample_weights, languages_val, verbosity, maxlen, format, save_every, filename):

    training = Probing()

    training.add_pretrained_model(model, dmap, copy_weights=['recurrent', 'embeddings'], train_classifier=True, train_embeddings=False, train_recurrent=False, mask_zero=True, classifiers=classifiers, dropout_recurrent=dropout_recurrent, optimizer=optimizer)

    dmap = pickle.load(open(dmap, 'rb'))

    X_train, Y_train = generate_training_data(training, languages_train, dmap, digits, format, classifiers, maxlen)

    validation_data = None
    if languages_val:
        X_val, Y_val = generate_training_data(training, languages_val, dmap, digits, format, classifiers, maxlen)
        validation_data = (X_val, Y_val)

    training.train((X_train, Y_train), batch_size=batch_size, epochs=nb_epochs, validation_split=validation_split, validation_data=validation_data, sample_weight=sample_weights, verbosity=verbosity, save_every=save_every, filename=filename)

    return training

def test_model(training, languages_test, dmap, digits, maxlen, test_separately, classifiers, sample_weights, format):
    """
    Test model on new test set, plot also the training and validation
    error the model had during training
    :param training:         trainings object
    :param languages_test:   dictionary mapping language names to number of items  
    """

    # open dmap
    dmap = pickle.load(open(dmap, 'rb'))

    # generate test data
    test_data = generate_test_data(Probing(), languages=languages_test, dmap=dmap,
                                   digits=digits, pad_to=maxlen, classifiers=classifiers,
                                   test_separately=test_separately, format=format)

    hist = training.trainings_history

    try:
        print "\nAccuracy for for training set %s:\t" % \
            '\n'.join(['\t%s:\t %s' % (output, '\t'.join('%s: %f' % (name_metric, hist.metrics_train[output][name_metric][-1]) for name_metric in hist.metrics[output])) for output in hist.metrics_train])
    except TypeError:
        print "typeerror"
        print hist.metrics_train
        '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_train.items()])
            
            
    try:
        print "Accuracy for for validation set %s:\t" % \
            '\n'.join(['%s:\t %s' % (output, '\t'.join('%s: %f' % (name_metric, hist.metrics_val[output][name_metric][-1]) for name_metric in hist.metrics[output])) for output in hist.metrics_val])
    except TypeError:
        '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_val.items()])


        # '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_val.items()])
    for name, X, Y in test_data:
        acc = training.model.evaluate(X, Y, sample_weight=sample_weights)
        print "\nAccuracy for for test set %s:" % name,
        print '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(len(acc))])


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

    training = train_model(model=settings.model, dmap=settings.dmap, classifiers=settings.classifiers, digits=settings.digits, languages_train=settings.languages_train,  optimizer=settings.optimizer, dropout_recurrent=settings.dropout_recurrent, batch_size=settings.batch_size, nb_epochs=settings.nb_epochs, validation_split=settings.validation_split, sample_weights=settings.sample_weights, languages_val=settings.languages_val, verbosity=settings.verbosity, maxlen=settings.maxlen, format=settings.format, save_every=settings.save_every, filename=settings.filename)

    if settings.languages_test:
        test_model(training, settings.languages_test, settings.dmap, settings.digits, settings.maxlen, settings.test_separately, settings.classifiers, sample_weights=settings.sample_weights, format=settings.format)

    # save model
    save_model(training)
