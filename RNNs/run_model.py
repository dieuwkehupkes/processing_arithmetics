from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from keras.utils.layer_utils import layer_from_config
from keras import backend as K
from generate_training_data import generate_test_data
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
# model.layers[2].return_sequences = True
model.compile(optimizer=settings.optimizer, loss=settings.loss, metrics=settings.metrics)

dmap = pickle.load(open(settings.model_dmap, 'rb'))
maxlen = model.layers[2].input_shape[1]

# TODO dmap should be identical for model and testset, currently there
# TODO seems to be no way to check this? Maybe I should make my own model class
# check if test sets are provided or should be generated
if isinstance(settings.test_sets, dict):
    test_data = generate_test_data(settings.test_sets, architecture='A1', dmap=dmap,
                                   digits=settings.digits, pad_to=maxlen)

elif isinstance(settings.test_sets, list):
    test_data = []
    for filename in settings.test_sets:
        X_val, Y_val = pickle.load(open(filename, 'rb'))
        test_data.append((filename, X_val, Y_val))

else:
    print("Invalid format test data")

# compute overall accuracy
if settings.compute_accuracy:
    metrics = model.metrics_names
    for name, X_test, Y_test in test_data:
        print "Accuracy for %s\t" % name,
        acc = model.evaluate(X_test, Y_test, verbose=0)
        print '\t'.join(['%s: %f' % (metrics[i], acc[i]) for i in xrange(len(acc))])

if settings.compute_correls:

    # check embeddings layer type:
    layer_type = {'SimpleRNN': SimpleRNN, 'GRU': GRU, 'LSTM': LSTM}
    recurrent_layer = layer_type[model.get_config()['layers'][2]['class_name']]
    rec_config = model.layers[2].get_config()

    embeddings_sequence = recurrent_layer(output_dim=rec_config['output_dim'],
                                          activation=rec_config['activation'],
                                          weights=model.layers[2].get_weights(),
                                          return_sequences=True)(model.layers[1].output)

    truncated_model = Model(input=model.layers[0].input, output=embeddings_sequence)
    truncated_model.compile(optimizer=settings.optimizer, loss=settings.loss, metrics=settings.metrics)

    for name, X_test, Y_test in test_data:
        predictions = truncated_model.predict(X_test)
        non_zero = predictions[np.any(predictions!=0, axis=2)]
        print np.corrcoef(non_zero)


    # Compute correlation between hidden unit activations
    # by truncating the model at the recurrent layer and
    # running predict computing all output activations
    #

