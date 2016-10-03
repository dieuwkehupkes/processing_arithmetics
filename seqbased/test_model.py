from __future__ import print_function
import sys 
sys.path.insert(0, '../commonFiles') 
import keras.preprocessing.sequence
from keras.models import Model, load_model
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from analyser import visualise_hidden_layer
from architectures import A1, A4, Probing, Training
from arithmetics import mathTreebank
from train_model import generate_test_data
import argparse
from collections import OrderedDict, defaultdict
import pickle
import re
import numpy as np

def test_model(architecture, model, dmap, optimizer, loss, metrics, digits, test_sets, classifiers, test_separately=True, print_results=True, format='infix', crop_to_length=True):
    # load model
    architecture = architecture()
    dmap = pickle.load(open(dmap, 'rb'))
    compiled_model = load_model(model)

    maxlen = compiled_model.layers[2].input_shape[1]
    test_data = generate_test_data(architecture=architecture, languages=test_sets, dmap=dmap, digits=digits, maxlen=maxlen, test_separately=test_separately, classifiers=classifiers, format=format)

    metrics = compiled_model.metrics_names
    test_results = dict([(metric, OrderedDict()) for metric in metrics])
    for name, X_test, Y_test in test_data:
        if crop_to_length == True:
            length = re.compile('[0-9]+')
            n = int(length.search(name).group())
            maxlen = 4*n-3 
            architecture.add_pretrained_model(compiled_model, dmap, input_length=maxlen)
            compiled_model = architecture.model

            for key in X_test:
                X_test[key] = keras.preprocessing.sequence.pad_sequences(X_test[key], maxlen=maxlen)
            for key in Y_test:
                # check whether output needs to be padded
                if Y_test[key].ndim == 1:
                    continue
                Y_test[key] = keras.preprocessing.sequence.pad_sequences(Y_test[key], maxlen=maxlen)

        acc = compiled_model.evaluate(X_test, Y_test, verbose=0)
        if print_results:
            print("Accuracy for %s\t" % name, end=" ")
            print('\t'.join(['%s: %f' % (metrics[i], acc[i]) for i in xrange(len(acc))]))
        # store test results
        for i in xrange(len(acc)):
            try:
                test_results[metrics[i]][name] = acc[i]
            except KeyError:
                assert metrics[i] == 'loss'

    return test_results

def print_test_results(test_results):
    # Print test results in readable way
    raise NotImplementedError


if __name__ == '__main__':

    # import settings file
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="Provide file with settings for model running")

    args = parser.parse_args()

    import_string = args.settings
    py = re.compile('\.py$')
    if py.search(import_string):
        # take off .py
        import_string = import_string[:-3]

    settings = __import__(import_string)

    results = dict()
    for model in settings.models:
        print("\ntesting model ", model)
        model_dmap = settings.dmap

        results_model = test_model(architecture=settings.architecture, model=model, dmap=model_dmap, optimizer=settings.optimizer, metrics=settings.metrics, loss=settings.loss, digits=settings.digits, test_sets=settings.test_sets, classifiers=settings.classifiers, test_separately=settings.test_separately, format=settings.format, crop_to_length=settings.crop_to_length)

        results[model] = results_model

    per_language = defaultdict(list)
    for model in settings.models:
        for language in results[model]['loss']:
            per_language[language].append(results[model]['loss'][language])

    print('means:')
    for language in per_language:
        print(language, '\t\t', np.mean(per_language[language]))

    pickle.dump(results, open('results_dicts.pickle', 'wb'))

