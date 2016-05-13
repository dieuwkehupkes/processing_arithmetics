from keras.models import model_from_json
import argparse
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument("settings", help="Provide file with settings for model running")

args = parser.parse_args()

import_string = args.settings
py = re.compile('\.py$')
if py.search(import_string):
    # take of .py
    import_string = import_string[:-3]

settings = __import__(import_string)

# load model
model = model_from_json(open(settings.model_architecture).read())
model.load_weights(settings.model_weights)
model.compile(optimizer=settings.optimizer, loss=settings.loss, metrics=settings.metrics)

dmap = pickle.load(open(settings.model_dmap, 'rb'))

# TODO this might be different now, check nr of objects in this file
# TODO also, dmap should be identical for model and testset, currently there
# TODO seems to be no way to check this? Maybe I should make my own model class
X_val, Y_val = pickle.load(open(settings.test_set, 'rb'))

# compute overall accuracy
if settings.compute_accuracy:
    acc = model.evaluate(X_val, Y_val)
    metrics = model.metrics_names
    print '\n'.join(['%s: %f' % (metrics[i], acc[i]) for i in xrange(len(acc))])
