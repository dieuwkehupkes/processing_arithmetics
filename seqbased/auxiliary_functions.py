"""
Collection of auxiliary functions.

Describe functions?
"""

import numpy as np


# function to print summary of results
def print_sum(settings):
    print('Model summary:')
    print('Recurrent layer: %s' % str(settings.recurrent_layer))
    print('Size hidden layer: %i' % settings.size_hidden)
    print('Initialisation embeddings: %s' % settings.encoding)
    print('Size embeddings: %i' % settings.input_size)
    print('Batch size: %i' % settings.batch_size)
    print('Number of epochs: %i' % settings.nb_epoch)
    print('Optimizer: %s' % settings.optimizer)
    print('Trained on:')
    try:
        for language, nr in settings.languages_train.items():
            print('%i sentences from %s' % (nr, language))
    except AttributeError:
        print('Unknown')


def save_model(training):
    """
    Save model to file.
    """
    import pickle
    import os
    save = raw_input("\nSave model? y/n ")
    if save == 'n' or save == 'N':
        pass
    elif save == 'y' or save == 'Y':
        exists = True
        while exists:
            model_string = raw_input("Provide filename (without extension) ")
            exists = os.path.exists(model_string + '_weights.h5')
            if exists:
                overwrite = raw_input("File name exists, overwrite? (y/n) ")
                if overwrite == 'y':
                    exists = False
                continue

        model_json = training.model.to_json()
        open(model_string + '.json', 'w').write(model_json)
        training.model.save_weights(model_string + '_weights.h5')
        hist = training.trainings_history
        trainings_history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
        pickle.dump(trainings_history, open(model_string + '.history', 'wb'))
    return

def max_length(N):
    """
    Compute length of arithmetic expression
    with N numeric leaves
    :param N: number of numeric leaves of expression
    """
    l = 4*N-3
    return l
