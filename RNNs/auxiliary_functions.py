"""
Collection of auxiliary functions.

Describe functions?
"""


import numpy as np
import theano.tensor as T

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
        assert input_size == N_digits + N_operators + 2, "Identity encoding not possible if input size does not equal input dimension" 
        return [np.identity(input_size)]

    # return GrayCode, raise exception if input_size is too small
    if encoding == 'Gray' or encoding == 'gray':
        return [grayEmbeddings(N_digits, N_operators, input_size)]


def grayEmbeddings(N_digits, N_operators, input_size):
    """
    Generate embeddings where numbers are encoded with
    reflected binary code. both embeddings and brackets are
    randomly initialised with values between -0.1 and 0.1
    """
    # generate graycode for digits
    grayDigits = grayCode(N_digits, input_size)

    # extend with random vectors for operators and brackets
    grayDigits.extend([np.random.random_sample(input_size)*.2-.1 for i in xrange(N_operators+2)])

    return np.array(grayDigits)


def grayCode(n, length=None):
    grays = [[0.0],[1.0]]
    while len(grays) < n+1:
        pGrays = grays[:]
        grays = [[0.0]+gray for gray in pGrays]+[[1.0]+gray for gray in pGrays[::-1]]

    # reduce to length n
    grays = grays[1:n+1]

    # pad to right length
    if length:
        gray_code = [pad(gray, length) for gray in grays]
    else:
        gray_code = [gray for gray in grays]
    return gray_code

if __name__ == '__main__':
    print grayCode(10, 7)

