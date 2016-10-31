"""

"""


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

def max_length(N):
    """
    Compute length of arithmetic expression
    with N numeric leaves
    :param N: number of numeric leaves of expression
    """
    l = 4*N-3
    return l
