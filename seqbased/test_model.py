from __future__ import print_function
import sys 
sys.path.insert(0, '../commonFiles') 
from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from analyser import visualise_hidden_layer
from architectures import *
import argparse
import pickle
import re
import numpy as np

def test_model(architecture, model_architecture, model_weights, dmap, optimizer, loss, metrics, digits, test_sets, test_separately=True):
    # load model
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    dmap = pickle.load(open(model_dmap, 'rb'))
    dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])
    maxlen = model.layers[2].input_shape[1]

    # TODO dmap should be identical for model and testset, currently there
    # TODO seems to be no way to check this? Maybe I should make my own model class
    # check if test sets are provided or should be generated
    if isinstance(test_sets, dict):
        test_data = architecture.generate_test_data(test_sets, dmap=dmap,
                                                             digits=digits, pad_to=maxlen,
                                                             test_separately=test_separately)

    elif isinstance(test_sets, list):
        test_data = []
        for filename in test_sets:
            X_val, Y_val = pickle.load(open(filename, 'rb'))
            test_data.append((filename, X_val, Y_val))

    else:
        print("Invalid format test data")

    # compute overall accuracy
    metrics = model.metrics_names
    for name, X_test, Y_test in test_data:
        print("Accuracy for %s\t" % name, end=" ")
        acc = model.evaluate(X_test, Y_test, verbose=0)
        print('\t'.join(['%s: %f' % (metrics[i], acc[i]) for i in xrange(len(acc))]))

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

    for pref in settings.prefs:
        print("\ntesting model ", pref)
        model_architecture = pref+'.json'         # name of file containing model architecture
        model_weights = pref+'_weights.h5'     # name of file containing model weights
        model_dmap = pref+'.dmap'              # dmap of the embeddings layer of the model

        test_model(architecture=settings.architecture, model_architecture=model_architecture, model_weights=model_weights, dmap=model_dmap, optimizer=settings.optimizer, metrics=settings.metrics, loss=settings.loss, digits=settings.digits, test_sets=settings.test_sets, test_separately= settings.test_separately)
