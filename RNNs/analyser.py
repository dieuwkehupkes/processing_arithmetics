"""
Collection of functions to analyse the weight matrices of
the network.
"""

import itertools as it
import numpy as np
import matplotlib.pylab as plt
import matplotlib

# def visualise_hidden_layer(*inputs):
#     """
#     Plot hidden layer activations over time
#     :param inputs:  tuple with lists containing hl activations and labels
#     """
#     col = 1
#     rows = 2*len(inputs)
#     plt.figure(1)
#     for input in inputs:
#         hl_activations, labels = input
#         # cut off zero activations
#         if len(hl_activations.shape) == 3:
#             hl_nonzero = hl_activations[np.any(hl_activations!=0, axis=2)]
#         else:
#             hl_nonzero = hl_activations
#         vmin = np.min(hl_activations)
#         vmax = np.max(hl_activations)
#
#         l = hl_nonzero.shape[0]
#         for i in xrange(l):
#             # plot label
#             #plt.subplot(l, rows, rows*i+col)
#             #plt.text(0,0, labels[i])
#             #plt.axis([-0.1, 0.1, -0.1, 0.1])
#
#             # plot activation values
#             plt.subplot(l, rows, rows*i+col)
#             plt.imshow(np.array([hl_nonzero[i]]), interpolation='nearest', cmap=colourmap(), vmin=vmin, vmax=vmax)
#             plt.text(-2, -1, labels[i])
#             plt.axis('off')
#
#         col += 1
#     plt.subplots_adjust(bottom=0.1, right=0.7, top=0.9, wspace=0.1, hspace=0.0)
#     #plt.tight_layout(h_pad=0.2)
#     cax = plt.axes([0.85, 0.1, 0.025, 0.8])
#     plt.colorbar(cax=cax)
#     plt.show()

def visualise_hidden_layer(*inputs):
    """
    Plot hidden layer activations over time
    :param hl_activations: 2 dimensional np array

    """

    cols = len(inputs)
    col = 1

    maxlen = find_longest(inputs)

    fig = plt.figure(figsize=(9*cols, maxlen/2))
    for hl_activations, labels in inputs:

        # cut off zero activations
        if len(hl_activations.shape) == 3:
            hl_nonzero = hl_activations[np.any(hl_activations!=0, axis=2)]
        else:
            hl_nonzero = hl_activations
        vmin = -1
        vmax = 1

        rows = hl_nonzero.shape[0]
        plt.subplot(1, cols, col)
        plt.imshow(hl_nonzero, interpolation='nearest', cmap=colourmap(), vmin=vmin, vmax=vmax)
        for i in xrange(rows):
            # plot activation values and labels
            plt.text(-2, i, labels[i])
            plt.axis('off')
        col += 1

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.1, 0.025, 0.8])
    plt.colorbar(cax=cax, orientation='vertical')
    plt.show()


def find_longest(inputs):
    """
    Find longest sequence in input parameters
    :param inputs: sequence of tuples (input_sequence, labels)
    :return: integer describing the longest sequence within the inputs
    """
    maxlen = 1
    for input in inputs:
        maxlen = max(maxlen, len(input[1]))

    return maxlen



def distance_embeddings(embeddings_matrix):
    """
    Compute the average distance between embeddings
    """
    distance = 0
    l = 0
    for row1, row2 in it.combinations(embeddings_matrix, 2):
        new_distance = np.sqrt(np.sum(np.power(row1-row2, 2)))
        distance += new_distance
        l += 1

    av_distance = distance / l
    return av_distance


def plot_distances(embeddings_matrix):
    """
    Create a plot of the distances between the different
    vectors in the embeddings matrix.
    """
    distances = []
    for row1, row2 in it.combinations(embeddings_matrix, 2):
        distance = np.sqrt(np.sum(np.power(row1-row2, 2)))
        distances.append(distance)

    plt.hist(distances, bins = len(distances)/50)
    plt.xlabel("Distance between vectors")
    plt.ylabel("Number of vectors")
    plt.show()
    return


def length_embeddings(embeddings_matrix):
    """
    Compute the average length of the embeddings.
    """
    length = 0
    l = len(embeddings_matrix)
    for embedding in embeddings_matrix:
        length += np.sqrt(np.sum(np.power(embedding, 2)))

    av_length = length/l
    
    return av_length

def colourmap():
    cdict = {'red':     ((0.0, 0.0, 0.0),
                         (0.1, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (0.9, 1.0, 1.0),
                         (1.0, 0.7, 0.7)),

            'green':    ((0.0, 0.0, 0.0),
                         (0.1, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (0.9, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),

            'blue':     ((0.0, 0.7, 0.7),
                         (0.1, 1.0, 1.0),
                         (0.5, 1.0, 1.0),
                         (0.9, 0.0, 0.0),
                         (1.0, 0.0, 0.0))
             }
    colourmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return colourmap

def colourmap1():
    cdict = {'red': ((0.0, 1.0, 1.0), (1.0, 0.5, 0.5)),
             'green': ((0.0, 1.0, 1.0), (1.0, 0.5, 0.5)),
             'blue': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))}
    colourmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return colourmap


def colourmap2():
    cdict = {'red': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
             'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
             'blue': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}
    colourmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return colourmap
