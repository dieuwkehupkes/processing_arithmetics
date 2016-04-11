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

    def generate_update_function(self):
        """
        Generate a symbolic graph describing what happens when the 
        network gets updated
        """

        # generate symbolic variable for current input
        input_t = T.vector("input_t")

        # TODO change this so that it uses an activation function set as input
        hidden_next = piecewise_linear(T.dot(input_t, self.U) + T.dot(self.activations['hidden_t'], self.V) + self.b1)

        # define the next hidden and output state by applying the network dynamics
        # use intermediate steps to model behaviour
        # use piecewise linear activation functions
        hidden_project1 = T.dot(input_t, self.U)
        hidden_project2 = T.dot(self.activations['hidden_t'], self.V)
        hidden_sum = hidden_project1 + hidden_project2 + self.b1
        hidden_squash = piecewise_linear(hidden_sum)
        # hidden_next = T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(self.activations['hidden_t'], self.V) + self.b1)
        
        output_project = T.dot(self.activations['hidden_t'], self.W)
        output_sum = output_project + self.b2
        output_next = T.nnet.sigmoid(T.dot(self.activations['hidden_t'], self.W) + self.b2)
        # output_next = piecewise_linear(T.dot(self.activations['hidden_t'], self.W) + self.b2)

        # prediction is defined by output layer
        # prediction = self.prediction(output_next)

        # define function that executes forward pass
        updates = OrderedDict(zip(self.activations.values(), [hidden_next, output_next]))

        # define functions to visualise project squash sum
        self.forward_pass = theano.function([input_t], updates=updates, givens={})
        self.project_input = theano.function([input_t], hidden_project1, givens={})
        self.project_hidden = theano.function([], hidden_project2, givens={})
        self.sumh = theano.function([input_t], hidden_sum, givens={})
        self.print_hidden = theano.function([input_t], hidden_next, givens={})
        self.squash = theano.function([input_t], squash, givens={})

        # function that returns current prediction
        # self.cur_prediction = theano.function([], prediction)

        return

    def generate_network_dynamics(self):
        """
        Create symbolic expressions defining how the network behaves when
        given a sequence of inputs, and how its parameters can be trained.
        In this function it is defined:
        - How a sequence of hidden-layer activations can be computed
          from an input sequence
        - How a sequence of predictions can be computed from a sequence
          of hidden layer activations
        - How the next-item prediction of the network can be computed
        - How the error of a sequence of predictions can be computed from
          the input sequence
        - How to compute the gradients w.r.t the different weights
        """

        # Set parameters to be updated during training and initialise adagrad parameters
        # default to be updated: W, U, V, b1 and b2
        self.set_network_parameters()

        # create symbolic variable for input sequence (batch processing)
        input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)      # tensor 3-dim
        input_sequences_transpose = input_sequences.transpose(1,0,2)                    # tensor 3-dim

        # describe how the hidden layer can be computed from the input
        def calc_hidden(input_map_t, hidden_t):
            return T.nnet.sigmoid(T.dot(input_map_t, self.U) + T.dot(hidden_t, self.V) + self.b1)

        hidden_t = T.matrix("hidden_t", dtype=theano.config.floatX)                     # tensor 2-dim

        # compute sequence of hidden layer activations for all input sequences by using scan
        hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences_transpose, outputs_info=hidden_t)      # tensor 3-dim

        # compute sequence of output layer activations (don't include last one which is the prediction after the end of the string)
        # output_sequences = T.nnet.sigmoid(T.dot(hidden_sequences, self.W) + self.b2)[:-1]    # tensor 3-dim
        output_sequences = piecewise_linear(T.dot(hidden_sequences, self.W) + self.b2)[:-1]    # tensor 3-dim


        # TODO include function that says which part of the predictions should be trained/tested

        # compute predictions and target predictions
        predictions = T.argmax(output_sequences, axis = 1)              # TODO test if this does what I want
        predictions_last = T.argmax(output_sequences_last, axis = 1)
        target_predictions = T.argmax(input_sequences_transpose[1:], axis = 1)      # TODO test if this does what I want
        target_predictions_last = T.argmax(input_sequences_transpose[-1], axis = 1)

        # compute sum squared differences between predicted numbers
        spe = T.sqr(predictions - target_predictions)   # squared prediction error per item
        sspe = T.sum(spe)           # sum squared prediction error
        mspe = T.mean(spe)           # mean squared prediction error
        spel = T.sqr(predictions_last - target_predictions_last)   # squared prediction error of last element per item
        sspel = T.sum(spel)                                # sum squared prediction error of last element
        mspel = T.mean(spe)                                # mean squared prediction error of last element

        # compute the difference between the output vectors and the target output vectors
        errors = T.nnet.categorical_crossentropy(output_sequences[-1], input_sequences_transpose[-1])
        error = T.mean(errors)

        # compute gradients
        grads = T.grad(error, self.params.values())
        gradients = OrderedDict(zip(self.params.keys(), grads))

        theano.pp(gradients['W'])

        # compute new parameters
        new_params = OrderedDict()
        for param in self.params:
            new_histgrad = self.histgrad[param] + T.sqr(gradients[param])
            new_param_value = self.params[param] - self.learning_rate*gradients[param]/(T.sqrt(new_histgrad) + 0.000001)
            new_params[self.params[param]] = new_param_value
            new_params[self.histgrad[param]] = new_histgrad

        # initial values
        givens = {
                hidden_t:       T.zeros((input_sequences_transpose.shape[1], self.hidden_size)).astype(theano.config.floatX),
        }


        self.print_grad_W = theano.function([input_sequences], gradients['W'], givens=givens)

        # define functions

        # run update function to train weights
        self.update_function = theano.function([input_sequences], updates=new_params, givens=givens)

        # compute the differences of the output vectors with the target vectors
        self.compute_error = theano.function([input_sequences], errors, givens=givens)

        # compute the differences of the meaning of the output vectors with the target meanings
        # TODO???
        self.prediction_error_diff = theano.function([input_sequences], T.sqrt(spel), givens=givens)

        # take the sum of the latter for the whole batch
        self.sum_squared_prediction_error = theano.function([input_sequences], sspe, givens=givens)
        self.mean_squared_prediction_error = theano.function([input_sequences], mspe, givens=givens)
        self.sum_squared_prediction_last_error = theano.function([input_sequences], sspel, givens=givens)
        self.mean_squared_prediction_last_error = theano.function([input_sequences], mspel, givens=givens)

        # print network predictions for current batch
        self.predictions = theano.function([input_sequences], predictions, givens=givens)

        # print target predictions for current batch
        self.target_predictions = theano.function([input_sequences], target_predictions, givens=givens)

        # temp functions for monitoring
        self.print_input_map_transpose = theano.function([input_sequences], input_sequences_map_transpose)
        self.print_hidden = theano.function([input_sequences], hidden_sequences, givens=givens)
        self.print_output = theano.function([input_sequences], output_sequences, givens=givens)
        self.print_predictions = theano.function([input_sequences], predictions, givens=givens)
        self.print_target_predictions = theano.function([input_sequences], target_predictions, givens=givens)

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

    def set_network_parameters(self, word_embeddings=None, classifier=None):
        # TODO This method should be moved to superclass!
        """
        If co training of word embeddings is desired,
        add word-embeddings matrix to trainable parameters
        of the network.
        """

        # store the parameters of the network
        self.params = OrderedDict([('U', self.U), ('V', self.V), ('W', self.W), ('b1', self.b1), ('b2', self.b2)])

        if word_embeddings:
            self.params['embeddings'] = self.embeddings

        if classifier:
            self.params['classifier'] = self.classifier

        # store history of gradients for adagrad
        histgrad = []
        for param in self.params:
            init_grad = np.zeros_like(self.params[param].get_value(), dtype=theano.config.floatX)
            name = "hist_grad_" + param
            histgrad.append((param, theano.shared(init_grad, name=name)))

        self.histgrad = OrderedDict(histgrad)

        return

    def output(self):
        output = self.activations['output_t'].get_value()
        return output

    def hidden(self):
        hidden = self.activations['hidden_t'].get_value()
        return hidden

