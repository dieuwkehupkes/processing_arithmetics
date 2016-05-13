from keras.models import model_from_json
from keras import backend as K
from generate_training_data import generate_test_data
import argparse
import pickle
import re

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
    # truncate model to output hidden layer activations
    # set return_sequences of hidden layer to true
    model.layers[2].return_sequences = True
    hl_activations = K.function([model.layers[0].input, K.learning_phase()],
                                model.layers[2].get_activations(train=False))

    for name, X_test, Y_test in test_data:
        # activations = hl_activations([X_test, 0])
        print hl_activations([X_test, 0])



    # Compute correlation between hidden unit activations
    # by truncating the model at the recurrent layer and
    # running predict computing all output activations
    #

