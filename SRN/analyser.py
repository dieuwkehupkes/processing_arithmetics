"""
Collection of functions to analyse the weight matrices of
the network.
"""

import itertools as it
import numpy as np
import matplotlib.pylab as plt


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
