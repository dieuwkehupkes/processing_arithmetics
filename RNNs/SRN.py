"""
Dieuwke Hupkes - <D.hupkes@uva.nl>
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from auxiliary_functions import *
from RNN import RNN

class SRN(RNN):
    """
    A class representing a simple recurrent network (SRN), as presented
    in Elman (1990).
    """
    def __init__(self, input_size, hidden_size, learning_rate=0.5, sigma_init=0.2):
        """
        This class implements a classic simple recurrent network in Theano:

            X_t = f(V*X_{t-1} + U*I_t)
            Y_t = g(W*X_t)

        where X_t is the hidden layer, Y_t is the output layer and f(x) is a
        simple sigmoid function.

        Describe the functionality implemented in this class

        :param input_size: number of input units
        :param hidden_size: number of hidden units
        :param sigma_init: parameter used to initialise network weights
        :param learning_rate: learning rate
        """

        RNN.__init__(
                self, 
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=input_size,
                learning_rate=learning_rate,
                sigma_init=sigma_init
        )

    def generate_forward_pass(self):
        """
        Generate a function that returns a batch of
        output sequences given a batch of input sequences.
        This corresponds with the forward pass of the
        network.
        """

        # create symbolic variable for input sequence (batch processing)
        input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)      # tensor 3-dim
        # transpose input sequence to loop over correct dimensions
        input_sequences_transpose = input_sequences.transpose(1,0,2)                    # tensor 3-dim

        # describe how the hidden layer can be computed from the input
        def calc_hidden(input_t, hidden_t):
            return T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(hidden_t, self.V) + self.b1)

        hidden_t = T.matrix("hidden_t", dtype=theano.config.floatX)                     # tensor 2-dim

        # compute sequence of hidden layer activations for all input sequences by using scan
        hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences_transpose, outputs_info=hidden_t)      # tensor 3-dim

        # compute sequence of output layer activations 
        output_sequences = T.nnet.sigmoid(T.dot(hidden_sequences, self.W) + self.b2)    # tensor 3-dim

        # initial values
        givens = {
                hidden_t:       T.zeros((input_sequences_transpose.shape[1], self.hidden_size)).astype(theano.config.floatX),
        }

        # return output sequences for input sequences
        self.return_outputs = theano.function([input_sequences], output_sequences, givens=givens)

        return

    def prediction(self, output_vector):
        """
        Compute the prediction of the network for output_vector
        by comparing its output with the word embeddings matrix.
        The prediction of the network will be the word embedding
        the output vector is closest to.
        NB: this function only works for an input *vector* and
        cannot be applied to an input matrix of vectors
        """
        # compute the distance with the embeddings
        e_distance_embeddings = T.sqrt(T.sum(T.sqr(self.embeddings-output_vector), axis=1))
        # prediction is embedding with minimal distance
        prediction = T.argmin(e_distance_embeddings)    
        return prediction

    def prediction_batch(self, output_matrix):
        """
        Compute the predictions of the network for a batch
        of output vectors.
        """
        raise NotImplementedError("Function not implemented yet")

    def set_network_parameters(self, word_embeddings=None):
        """
        Set trainable parameters of the network
        """

        # No extra trainable parameters required, use superclass method
        RNN.set_network_parameters(self, word_embeddings)

        return
