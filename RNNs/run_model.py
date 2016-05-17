from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from analyser import visualise_hidden_layer
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
dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])
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

compute_truncated = settings.compute_correls or settings.visualise_test_items

if compute_truncated:
    # generate model truncated at recurrent layer

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

if settings.compute_correls:
    # loop over test items
    for name, X_test, Y_test in test_data:
        predictions = truncated_model.predict(X_test)
        non_zero = predictions[np.any(predictions!=0, axis=2)]
        print np.corrcoef(non_zero)

if settings.visualise_test_items:
    # visualise test items one by one
    user_input = None
    i = 0
    for name, X_test, Y_test in test_data:
        predictions = model.predict(X_test)
        for s, m in zip(X_test, Y_test):
            while user_input != "q":
                labels = [dmap_inverted[word] for word in s[s.nonzero()]]
                test_item = ' '.join(labels)
                correct_prediction = str(m)
                model_prediction = round(predictions[i])
                print("Test item: %s\t\t Correct prediction: %s\t\t Model prediction: %i"
                      % (test_item, correct_prediction, model_prediction))
                hl_activations = truncated_model.predict(np.array([s]))
                visualise_hidden_layer(hl_activations, labels)

                user_input = raw_input()
                i += 1
            # visualise_next()


