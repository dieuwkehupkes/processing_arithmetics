from keras.models import Model, model_from_json
from keras.layers import Embedding, Input, GRU, LSTM, SimpleRNN, Dense
from analyser import visualise_hidden_layer
from GRU_output_gates import GRU_output_gates
from architectures import A1
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
dmap['x'] = 0
dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])
id = settings.architecture.get_recurrent_layer_id()
maxlen = model.layers[id].input_shape[1]
print maxlen


###########################################################################################
# GENERATE TEST DATA

# TODO dmap should be identical for model and testset, currently there
# TODO seems to be no way to check this? Maybe I should make my own model class
# check if test sets are provided or should be generated
if isinstance(settings.test_sets, dict):
    test_data = A1.generate_test_data(settings.test_sets, dmap=dmap,
                                                         digits=settings.digits, pad_to=maxlen)

elif isinstance(settings.test_sets, list):
    test_data = []
    for filename in settings.test_sets:
        X_val, Y_val = pickle.load(open(filename, 'rb'))
        test_data.append((filename, X_val, Y_val))

else:
    print("Invalid format test data")


###########################################################################################
# GENERATE MODEL TRUNCATED AT RECURRENT LAYER

# check embeddings layer type:
layer_type = {'SimpleRNN': SimpleRNN, 'GRU': GRU_output_gates, 'LSTM': LSTM}
recurrent_layer = layer_type[model.get_config()['layers'][id]['class_name']]
rec_config = model.layers[id].get_config()

embeddings_sequence = recurrent_layer(output_dim=rec_config['output_dim'],
                                      activation=rec_config['activation'],
                                      weights=model.layers[id].get_weights(),
                                      return_sequences=True)(model.layers[id-1].get_output_at(0))

truncated_model = Model(input=model.layers[0].input, output=embeddings_sequence)
truncated_model.compile(optimizer=settings.optimizer, loss=settings.loss, metrics=settings.metrics)

# print truncated_model.summary()


###########################################################################################
# COMPUTE CORRELATIONS BETWEEN HIDDEN UNITS

if settings.compute_correls:
    # loop over test items
    for name, X_test, Y_test in test_data:
        predictions = truncated_model.predict(X_test)
        non_zero = predictions[np.any(predictions!=0, axis=1)]
        print np.corrcoef(non_zero)


###########################################################################################
# VISUALISE PROJECTION OF LEXICAL ITEMS TO HIDDEN LAYER
if settings.project_lexical:
    output_dim = truncated_model.layers[-1].output_dim
    lex_size = len(dmap)
    if 0 in dmap.values():
        lex_size -= 1
    hl_activations = np.zeros(output_dim*lex_size).reshape(lex_size, output_dim)
    i = 0
    labels = []

    # we don't need this anymore later for training new models
    def conv(token):
        try:
            return int(token)
        except ValueError:
            return token

    for lex_item in sorted(dmap.keys(), key=conv):
        if dmap[lex_item] == 0:
            continue
        input_seq = np.array([[dmap[lex_item]]])
        seq_padded = keras.preprocessing.sequence.pad_sequences(input_seq, dtype='int32', maxlen=maxlen)
        hl_activation = truncated_model.predict(seq_padded)[:,:,:output_dim]
        hl_non_zero = hl_activation[np.any(hl_activation!=0, axis=2)]
        hl_activations[i] = hl_non_zero
        labels.append(lex_item)
        i+=1

    visualise_hidden_layer([hl_activations, labels])


###########################################################################################
# VISUALISE TEST ITEMS ONE BY ONE
if settings.one_by_one:
    output_dim = truncated_model.layers[-1].output_dim
    user_input = None
    i = 0
    hl_activations = []
    for name, X_test, Y_test in test_data:
        if settings.architecture == A1:
            predictions = model.predict(X_test)
            model_predictions = predictions.round()
        for s, m in zip(X_test, Y_test):
            # labels = [dmap_inverted[word] for word in s[s.nonzero()]]
            # find first non zero
            nz1 = s.nonzero()[0][0]
            labels = [dmap_inverted[word] for word in s[nz1:]]
            test_item = ' '.join(labels)
            correct_prediction = str(m)
            if settings.architecture == A1:
                model_prediction = str(model_predictions[i])
            else:
                model_prediction = "No scalar prediction for A4 architecture"
            print("Test item: %s\t\t Correct prediction: %s\t\t Model prediction: %s"
                  % (test_item, correct_prediction, model_prediction))
            outputs = truncated_model.predict(np.array([s]))
            hl_activation = outputs[:,:,:output_dim]
            z = outputs[:,:,output_dim:2*output_dim]
            r = outputs[:,:,2*output_dim:]
            new_input = (hl_activation, labels)
            hl_activations.append((new_input))
            i += 1

            if i % settings.one_by_one == 0:
                visualise_hidden_layer(*tuple(hl_activations))
                hl_activations = []

                user_input = raw_input("Press enter to continue, q to quit ")

                if user_input == "q":
                    exit()
