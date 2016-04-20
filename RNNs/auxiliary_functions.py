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


def generate_embeddings_matrix(input_dim, input_size, encoding):
    """
    Generate matrix that maps integers representing one-hot
    vectors to word-embeddings.
    :param input_length:    size of the input layer
    :param input_dim:       dimensionality of the one-hot vectors 
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
        assert input_size == input_dim, "Identity encoding not possible if input size does not equal input dimension" 
        return np.identity(input_size)

    # return GrayCode, raise exception if input_size is too small
    if encoding == 'Gray' or encoding == 'gray':
        return grayCode(input_dim, input_size)


def grayCode(n, length):
    grays = [[0.0],[1.0]]
    while len(grays) < n+1:
        pGrays = grays[:]
        grays = [[0.0]+gray for gray in pGrays]+[[1.0]+gray for gray in pGrays[::-1]]

    # pad to right length
    gray_code = np.array([pad(gray, length) for gray in grays])
    return gray_code

if __name__ == '__main__':
    print grayCode(10, 7)

