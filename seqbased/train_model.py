import sys 
sys.path.insert(0, '../commonFiles') 
import argparse
import pickle
from generate_training_data import generate_dmap
from auxiliary_functions import generate_embeddings_matrix, print_sum
from architectures import Training
from arithmetics import mathTreebank
import re


def train(architecture, languages_train, languages_val, validation_split, dmap, digits, format, classifiers, maxlen, pretrained_model, copy_weights, recurrent_layer, train_classifier, train_embeddings, train_recurrent, mask_zero, optimizer, dropout_recurrent, input_dim, input_size, size_hidden, batch_size, nb_epochs, verbose, sample_weights, save_every, filename):
    """
    Generate model and train.
    """
    training = architecture()

    training_data = generate_training_data(architecture=training, languages=languages_train, dmap=dmap, digits=digits, format=format, classifiers=classifiers, maxlen=maxlen)

    if languages_val:
        validation_data = generate_training_data(architecture=training, languages=languages_val, dmap=dmap, digits=digits, format=format, classifiers=classifiers, maxlen=maxlen)
    else:
        validation_data = None

    training = add_model(architecture=training, pretrained_model=pretrained_model, copy_weights=copy_weights, recurrent_layer=recurrent_layer, train_classifier=train_classifier, train_embeddings=train_embeddings, train_recurrent=train_recurrent, mask_zero=mask_zero, optimizer=optimizer, dropout_recurrent=dropout_recurrent, input_dim=input_dim, input_size=input_size, input_length=maxlen, size_hidden=size_hidden, classifiers=classifiers, sample_weights=sample_weights)

    training.train(training_data=training_data, validation_data=validation_data,
                   validation_split=validation_split, batch_size=batch_size,
                   epochs=nb_epochs, verbosity=verbose, filename=filename,
                   sample_weight=sample_weights, save_every=save_every)

    hist = training.trainings_history

    print "Accuracy for for training set %s:\t" % \
          '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_train.items()])
    print "Accuracy for for validation set %s:\t" % \
          '\t'.join(['%s: %f' % (item[0], item[1][-1]) for item in hist.metrics_val.items()])

    history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
    pickle.dump(history, open(filename + '.history', 'wb'))

    return training


def generate_training_data(architecture, languages, dmap, digits, format, classifiers, maxlen):
    """
    Generate training/validation data.
    """
    if isinstance(languages, dict):
        X, Y = architecture.generate_training_data(languages=languages, dmap=dmap,
                                               digits=digits, format=format,
                                               classifiers=classifiers,
                                               pad_to=maxlen)
    elif isinstance(languages, mathTreebank):
        X, Y = architecture.data_from_treebank(treebank=languages, dmap=dmap, pad_to=maxlen,
                                           classifiers=classifiers)

    return X, Y


def generate_test_data(architecture, languages, dmap, digits, maxlen, test_separately, classifiers):
    """
    Generate test data
    """

    if isinstance(languages, dict):
        test_data = Training.generate_test_data(architecture=training, 
                                                    languages=languages, dmap=dmap,
                                                    digits=digits, pad_to=settings.maxlen,
                                                    test_separately=settings.test_separately,
                                                    classifiers=classifiers)

    elif isinstance(languages, mathTreebank):
        X_test, Y_test = training.data_from_treebank(treebank, dmap=dmap, pad_to=settings.maxlen, classifiers=classifiers)
        test_data = [('test treebank', X_test, Y_test)]

    elif isinstance(languages, list):
        test_data = []
        for name, treebank in languages:
            X_test, Y_test = training.data_from_treebank(treebank, dmap=dmap, pad_to=settings.maxlen, classifiers=settings.classifiers)
            test_data.append((name, X_test, Y_test))

    return test_data


def add_model(architecture, pretrained_model, copy_weights, recurrent_layer, train_classifier, train_embeddings, train_recurrent, mask_zero, optimizer, dropout_recurrent, input_dim, input_size, input_length, size_hidden, classifiers, sample_weights):
    """
    Generate the trainings architecture and
    model to be trained.
    """

    if pretrained_model:
        model_string = pretrained_model + '.json'
        model_weights =pretrained_model + '_weights.h5'

        architecture.add_pretrained_model(model_string, model_weights, 
                                      dmap=dmap, copy_weights=copy_weights,
                                      train_classifier=train_classifier,
                                      train_embeddings=train_embeddings,
                                      train_recurrent=strain_recurrent,
                                      mask_zero=mask_zero,
                                      optimizer=optimizer,
                                      dropout_recurrent=dropout_recurrent)

    else:
        architecture.generate_model(recurrent_layer, input_dim=input_dim, input_size=input_size,
                                input_length=input_length, size_hidden=size_hidden,
                                dmap=dmap,
                                W_embeddings=None,
                                train_classifier=train_classifier, 
                                train_embeddings=train_embeddings,
                                train_recurrent=train_recurrent,
                                mask_zero=mask_zero,
                                optimizer=optimizer, dropout_recurrent=dropout_recurrent,
                                extra_classifiers=classifiers)

    return architecture

def test_model(training, languages_test, dmap, digits, maxlen, test_separately, classifiers):

    test_data = generate_test_data(architecture=training, languages=languages_test,
                                       dmap=dmap, digits=digits, maxlen=maxlen,
                                       test_separately=test_separately,
                                       classifiers=classifiers)

    for name, X, Y in test_data:
        acc = training.model.evaluate(X, Y)
        print "Accuracy for for test set %s:" % name,
        print '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(len(acc))])


if __name__ == '__main__':
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
    
    dmap = pickle.load(open('models/dmap', 'rb'))
    input_dim = len(dmap)+1

    training = train(architecture=settings.architecture, languages_train=settings.languages_train,
          languages_val=settings.languages_val, validation_split=settings.validation_split,
          dmap=dmap, digits=settings.digits, format=settings.format, maxlen=settings.maxlen,
          classifiers=settings.classifiers, pretrained_model=settings.pretrained_model,
          copy_weights=settings.copy_weights, recurrent_layer=settings.recurrent_layer,
          train_classifier=settings.train_classifier, train_embeddings=settings.train_embeddings,
          train_recurrent=settings.train_recurrent, mask_zero=settings.mask_zero,
          optimizer=settings.optimizer, dropout_recurrent=settings.dropout_recurrent,
          input_dim=input_dim, input_size=settings.input_size, sample_weights=settings.sample_weights,
          size_hidden=settings.size_hidden, save_every=settings.save_every, filename=settings.filename,
          batch_size = settings.batch_size, nb_epochs=settings.nb_epoch, verbose=settings.verbose)


    if settings.languages_test:
        test_model(training=training, languages_test=settings.languages_test, dmap=dmap,
                   digits=settings.digits, maxlen=settings.maxlen,
                   test_separately=settings.test_separately,
                   classifiers=settings.classifiers)

