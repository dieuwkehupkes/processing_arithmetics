from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from analyser import visualise_hidden_layer
from architectures import *
import argparse
import pickle
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running")

args = parser.parse_args()

import_string = args.settings
py = re.compile('\.py$')
if py.search(import_string):
    # take off .py
    import_string = import_string[:-3]

settings = __import__(import_string)

# load model
model = model_from_json(open(settings.model_architecture).read())
model.load_weights(settings.model_weights)
model.compile(optimizer=settings.optimizer, loss=settings.loss, metrics=settings.metrics)

dmap = pickle.load(open(settings.model_dmap, 'rb'))
dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])
maxlen = model.layers[2].input_shape[1]

# TODO dmap should be identical for model and testset, currently there
# TODO seems to be no way to check this? Maybe I should make my own model class
# check if test sets are provided or should be generated
if isinstance(settings.test_sets, dict):
    test_data = settings.architecture.generate_test_data(settings.test_sets, dmap=dmap,
                                                         digits=settings.digits, pad_to=maxlen)

elif isinstance(settings.test_sets, list):
    test_data = []
    for filename in settings.test_sets:
        X_val, Y_val = pickle.load(open(filename, 'rb'))
        test_data.append((filename, X_val, Y_val))

else:
    print("Invalid format test data")

# compute overall accuracy
metrics = model.metrics_names
for name, X_test, Y_test in test_data:
    print "Accuracy for %s\t" % name,
    acc = model.evaluate(X_test, Y_test, verbose=0)
    print '\t'.join(['%s: %f' % (metrics[i], acc[i]) for i in xrange(len(acc))])

