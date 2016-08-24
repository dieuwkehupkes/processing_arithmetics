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
    for language, nr in settings.languages_train.items():
        print('%i sentences from %s' % (nr, language))


# define padding function
def pad(array, length):
    l = len(array)
    try:
        l2 = len(array[0])
        print "length", l2
        padded_array = np.concatenate((np.zeros((length-l,l2)), array))
    except TypeError:
        try:
            padded_array = np.concatenate((np.zeros((length-l,)), array))
        except ValueError:
            raise ValueError("Array cannot be padded to shorter length")
    except ValueError:
        raise ValueError("Array cannot be padded to shorter length")
    return padded_array


def max_length(N):
    """
    Compute length of arithmetic expression
    with N numeric leaves
    :param N: number of numeric leaves of expression
    """
    l = 4*N-3
    return l


def generate_embeddings_matrix(N_digits, N_operators, input_size, encoding):
    """
    Generate matrix that maps integers representing one-hot
    vectors to word-embeddings.
    :param N_digits:        number of distinct digits in input
    :param N_operators:     number of operators
    :param input_size:      dimensionality of embeddings
    :param encoding:        gray or random or None
    More elaborate description of how this works?
    """
    # if encoding == random, let keras generate the embedding matrix
    if encoding == 'random':
        return None
    # if encoding == None, generate matrix that maps
    # input to one-hot vectors (this is only possible when
    # input_size == input_dim

    if encoding is None:
        assert input_size == N_digits + N_operators + 2,\
            "Identity encoding not possible if input size does not equal input dimension"
        return [np.identity(input_size)]

    # return GrayCode, raise exception if input_size is too small
    if encoding == 'Gray' or encoding == 'gray':
        return [gray_embeddings(N_digits, N_operators, input_size)]


def gray_embeddings(n_digits, n_operators, input_size):
    """
    Generate embeddings where numbers are encoded with
    reflected binary code. both embeddings and brackets are
    randomly initialised with values between -0.1 and 0.1
    """
    # generate gray code for digits
    gray_digits = gray_code(n_digits, input_size)

    # extend with random vectors for operators and brackets
    gray_digits.extend([np.random.random_sample(input_size) * .2 - .1 for i in xrange(n_operators + 2)])

    return np.array(gray_digits)


def gray_code(n, length=None):
    grays = [[0.0], [1.0]]
    while len(grays) < n+1:
        pgrays = grays[:]
        grays = [[0.0]+gray for gray in pgrays]+[[1.0]+gray for gray in pgrays[::-1]]

    # reduce to length n
    grays = grays[1:n+1]

    # pad to right length
    if length:
        gray_code = [pad(gray, length) for gray in grays]
    else:
        gray_code = [gray for gray in grays]
    return gray_code

if __name__ == '__main__':
    print gray_code(10, 7)

