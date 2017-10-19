"""
Collection of functions to analyse the weight matrices of
the network.
"""

import itertools as it
import numpy as np
# import matplotlib.cm as cm
# from matplotlib import gridspec
from sklearn.decomposition import PCA

def visualise_hidden_layer(output_classifier, *inputs):
    """
    Plot hidden layer activations over time
    :param hl_activations: 2 dimensional np array

    """
    import matplotlib.pylab as plt
    cols = len(inputs)
    col = 1

    maxlen = find_longest(inputs)

    # fig = plt.figure(figsize=(9*cols, maxlen/1.5))
    fig, axes = plt.subplots(nrows=2, ncols=len(inputs), sharex=True, gridspec_kw=dict(height_ratios=[maxlen, 1]))

    for hl_activations, labels, modes in inputs:

        # cut off zero activations
        hl_nonzero = hl_activations[np.any(hl_activations != 0, axis=-1)]
        
        rows = hl_nonzero.shape[0]
        plt.subplot(1, cols, col)
        plt.imshow(hl_nonzero, interpolation='nearest', cmap='bwr', vmin=-1, vmax=1)
        for i in xrange(rows):
            # plot activation values and labels
            plt.text(-1.3, i+0.2, labels[i])
            plt.text(-2.8, i+0.2, modes[i])
        plt.axis('off')
        col += 1

    fig.tight_layout(w_pad=0.8)

    # plt.subplots_adjust(right=0.9, bottom=0.1, top=0.9)
    # cax = plt.axes([0.95, 0.15, 0.02, 0.7])
    # plt.colorbar(cax=cax, orientation='vertical')
    plt.show()


def plot_gate_values(*inputs):
    """
    Plot hidden layer activations and gate values of GRU
    for processed sequences.
    :param inputs:
    :return:
    """
    # import matplotlib.pylab as plt
    rows, row = len(inputs), 1

    plt.subplots(nrows=len(inputs), ncols=3, sharex=True, sharey=True)

    for hl, z, r, labels, modes in inputs:

        # cut off zero activations
        hl_nonzero = hl[np.any(hl != 0, axis=-1)]
        z_nonzero = z[np.any(z != 0, axis=-1)]
        r_nonzero = r[np.any(r != 0, axis=-1)]

        length = hl_nonzero.shape[0]

        y_labels = ['%i\t%s'.expandtabs(2) % (modes[i], labels[i]) for i in xrange(length)]

        # plot hl activation
        hidden = plt.subplot(rows, 3, row*3-2)
        hidden.imshow(hl_nonzero, interpolation='nearest', cmap='bwr', vmin=-1, vmax=1)
        hidden.set_title("Hidden layer")

        # plot gate values
        update_gate = plt.subplot(rows, 3, row*3-1)
        update_gate.imshow(z_nonzero, interpolation='nearest', cmap=cm.get_cmap('Greys'), vmin=0, vmax=1)
        update_gate.set_title("Update gate")

        reset_gate = plt.subplot(rows, 3, row*3)
        reset_gate.imshow(r_nonzero, interpolation='nearest', cmap=cm.get_cmap('Greys'), vmin=0, vmax=1)
        reset_gate.set_title("Reset gate")

        row += 1 

        # set labels
        for ax in hidden, update_gate, reset_gate:
            ax.set_xticks([i for i in xrange(15)])
            ax.set_xticklabels([str(i) for i in xrange(1,16)])
            ax.set_yticks([i for i in xrange(length)])
            if ax == hidden:
                ax.set_yticklabels(y_labels, ha = 'left')
                ax.get_yaxis().set_tick_params(direction='out', pad=28, length=0)
            else:
                ax.set_yticklabels(['' for i in xrange(length)])


    # plt.subplots_adjust(right=0.9, bottom=0.1, top=0.9)
    # cax = plt.axes([0.95, 0.15, 0.02, 0.7])
    # plt.colorbar(cax=cax, orientation='vertical')

    plt.show()

def visualise_paths(*inputs):
    """
    Apply pca to hidden layer activations and visualise
    the paths through statespace in the reduced statespace
    :param inputs:
    :return:
    """
    import matplotlib.pylab as plt
    print(len(inputs))

    # create hl_activations matrix and compute principal components
    activations_nz =tuple([input[0][np.any(input[0]!=0, axis=2)] for input in inputs])
    hl_activations = np.concatenate(tuple(activations_nz))
    pca = PCA(n_components=2)
    pca.fit(hl_activations)

    for hl, z, r, labels in inputs:
        # cut off zero activations
        hl_nonzero = hl[np.any(hl!=0, axis=2)]
        description = ' '.join(labels)
        hl_reduced = pca.transform(hl_nonzero)
        plt.plot(hl_reduced[:,0], hl_reduced[:,1], label=description)
        # TODO I should maybe put some annotation of the trajectories, when are they where

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.legend()
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
    import matplotlib.pyplot as plt
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
