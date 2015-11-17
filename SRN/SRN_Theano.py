"""
Dieuwke Hupkes - <D.hupkes@uva.nl>
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


class SRN():
    """
    A class representing a simple recurrent network (SRN), as presented
    in Elman (1990).
    """
    def __init__(self, input_size, hidden_size, sigma_init):
        """
        The SRN is fully described by three weight matrices connecting
        the different layers of the network.

        :param input_size: number of input units
        :param hidden_size: number of hidden units
        """

        self.learning_rate = 0.01

        # weights from input to hidden
        self.U = theano.shared(
                value = np.random.normal(
                    0, sigma_init,
                    (input_size, hidden_size)
                ).astype(theano.config.floatX),
                name='U'
        )

        # weights from context to hidden
        self.V = theano.shared(
                value = np.random.normal(
                    0, sigma_init,
                    (hidden_size, hidden_size)
                ).astype(theano.config.floatX),
                name='V'
        )

        # weights from hidden to output
        self.W = theano.shared(
                value = np.random.normal(
                    0, sigma_init,
                    (hidden_size, input_size)
                ).astype(theano.config.floatX),
                name='W'
        )

        self.b1 = theano.shared(
                value = np.random.normal(
                    0, sigma_init, hidden_size
                ).astype(theano.config.floatX),
                name='b1'
        )

        self.b2 = theano.shared(
                value = np.random.normal(
                    0, sigma_init, input_size
                ).astype(theano.config.floatX),
                name = 'b2'
        )

        # Store the network activation values to run the network
        hidden_t = theano.shared(
                value = np.zeros(hidden_size).astype(theano.config.floatX),
                name = 'hidden_t'
        )
        output_t = theano.shared(
                value = np.zeros(input_size).astype(theano.config.floatX),
                name = 'output_t'
        )

        self.activations = OrderedDict(zip(['hidden_t','output_t'], [hidden_t, output_t]))

        # store dimensions of network
        self.input_size =  input_size
        self.hidden_size = hidden_size
        
        # store the parameters of the network
        self.params = OrderedDict([('U', self.U), ('V', self.V), ('W', self.W), ('b1', self.b1), ('b2', self.b2)])

        # store history of gradients for adagrad
        histgrad = []
        for param in self.params:
            init_grad = np.zeros_like(self.params[param].get_value(), dtype=theano.config.floatX)
            name = "hist_grad_" + param
            histgrad.append((param, theano.shared(init_grad, name=name)))

        self.histgrad = OrderedDict(histgrad)

    def generate_update_function(self):
        """
        Generate a symbolic expression describing how the network
        can be updated.
        """
        # current input and current hidden vector
        input_t = T.vector("input_t")
        hidden_t = T.vector("hidden_t")

        hidden_next = T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(self.activations['hidden_t'], self.V) + self.b1)

        output_next = T.flatten(T.nnet.softmax(T.dot(self.activations['hidden_t'], self.W) + self.b2))        # output_next = T.nnet.softmax(T.dot(self.activations['hidden_t'], self.W))

        updates = OrderedDict(zip(self.activations.values(), [hidden_next, output_next]))

        # givens = {}
        self.forward_pass = theano.function([input_t], updates=updates, givens={})

        return

    def generate_network_dynamics_batch(self):
        """
        Omschrijving.
        """
        
        # declare variables
        input_seqs = T.tensor3("input_seqs", dtype=theano.config.floatX)
        input_sequences = input_seqs.transpose(1,0,2)

        # describe how the hidden layer can be computed from the input
        def calc_hidden(input_t, hidden_t):
            # return hidden_t + 5
            return T.nnet.sigmoid(T.dot(input_t, self.U)  + T.dot(hidden_t, self.V) + self.b1)

        hidden_t = T.matrix("hidden_t", dtype=theano.config.floatX)

        # compute sequence of hidden layer activations
        hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences, outputs_info = hidden_t)
        # compute prediction sequence (i.e., output layer activation)
        output_sequences = self.softmax_tensor(T.dot(hidden_sequences, self.W) + self.b2)[:-1]       # predictions for all but last output

        # TODO print and test predictions

        # TODO print and test error
        errors = T.nnet.categorical_crossentropy(output_sequences, input_sequences[1:]) # vector
        error = T.mean(errors)  # scalar

        # TODO print and test prediction error

        # TODO compute and test gradients

        # TODO compute and test new parameters

        # initial values
        givens = {
                hidden_t:       T.zeros((input_sequences.shape[1], self.hidden_size)).astype(theano.config.floatX),
        }

        self.produce_output_batch = theano.function([input_seqs], output_sequences, givens=givens)
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


        NB: I somehow feel like this should not all be in the same class,
        but I am not really sure how to do it differently, because I
        also don't want to recreate expressions more often than necessary
        """

        # TODO Now this is a matrix describing an input sequence, ideally,
        # we would want this to be a vector of matrices describing input
        # sequences
        input_sequence = T.matrix("input_sequence", dtype=theano.config.floatX)

        # describe how the hidden layer can be computed from the input
        def calc_hidden(input_t, hidden_t):
            return T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(hidden_t, self.V) + self.b1)

        hidden_t = T.vector("hidden_t", dtype=theano.config.floatX)

        # compute sequence of hidden layer activations
        hidden_sequence, _ = theano.scan(calc_hidden, sequences=input_sequence, outputs_info = hidden_t)

        # compute prediction sequence (i.e., output layer activation)
        output_sequence = T.nnet.softmax(T.dot(hidden_sequence, self.W) + self.b2)[:-1]       # predictions for all but last output

        predictions = self.prediction(output_sequence)

        # symbolic definition of error
        errors = T.nnet.categorical_crossentropy(output_sequence, input_sequence[1:])   # vector
        error = T.mean(errors)      # scalar

        # prediction error, compute by dividing the number of correct predictions
        # by the total number of predictions
        prediction_errors = T.eq(predictions, self.prediction(input_sequence[1:]))
        prediction_error = 1 - T.mean(T.eq(predictions, self.prediction(input_sequence[1:])))   # scalar
        prediction_last = 1 - prediction_errors[-1]

        # gradients
        gradients = OrderedDict(zip(self.params.keys(), T.grad(error, self.params.values())))

        # updates for weightmatrices and historical grads
        new_params = []
        for param in self.params:
            new_histgrad = self.histgrad[param] + T.sqr(gradients[param])
            new_param_value = self.params[param] - gradients[param]/(T.sqrt(new_histgrad) + 0.000001)
            new_params.append((self.params[param], new_param_value))
            new_params.append((self.histgrad[param], new_histgrad))

        # initial values
        givens = {
                hidden_t:       np.zeros(self.hidden_size).astype(theano.config.floatX),
        }

        # function to update the weights
        self.update_function = theano.function([input_sequence], updates=new_params, givens=givens)     # update U, V, W, b1, b2

        # function to compute the cross-entropy error of the inputsequences
        self.compute_error = theano.function([input_sequence], error, givens=givens)

        # function for the prediction error on the entire sequence
        self.compute_prediction_error = theano.function([input_sequence], prediction_error, givens=givens)

        # prediction error only on the last elements of the sequences
        self.predict_last = theano.function([input_sequence], prediction_last, givens=givens)

        self.produce_output = theano.function([input_sequence], output_sequence, givens=givens)

        return

    def train(self, input_sequences, no_iterations, batchsize, some_other_params=None):
        """
        Train the network to store input_sequences
        :param input_sequences  
        :param no_iterations    
        """
        #TODO write function description
        for iteration in xrange(0, no_iterations):
            self.iteration(input_sequences, batchsize)

        return

    def iteration(self, input_sequences, batchsize):
        """
        Slice data in minibatches and perform one
        training iteration.
        :param input_sequences: The sequences we want to
                                store in the network
        """
        batches = self.make_batches(input_sequences, batchsize)

        # loop over minibatches, update parameters
        for batch in batches:
            self.update_function(batch[0]) 

        return

    def iteration_batch(self, input_sequences, batchsize):
        """
        Slice data in minibatches and perform one
        training iteration.
        :param input_sequences: The sequences we want to
                                store in the network
        """
        batches = self.make_batches(input_sequences, batchsize)

        # loop over minibatches, update parameters
        for batch in batches:
            self.update_function(batch) 

        return

    def make_batches(self, input_sequences, batchsize):
        """
        Make batches from input sequence. 
        Currently this doesn't do anything but return the
        input sequence (for testing phase) but later this
        should start doing some more things.
        """
        # TODO Make that this method actually does something
        # return permutated version of input sequences
        input_perm = np.random.permutation(input_sequences)
        return [input_perm]

    def prediction(self, output_vector):

        # assuming a 1-k encoding, the network prediction is the output unit
        # with the highest activation value after applying the sigmoid
        # If we are using distributed representations oid (what in the end
        # possibly desirable is, the symbolic expression for the prediction
        # for the prediction would change (as well as the output activation, btw)
        prediction = T.argmax(output_vector, axis=1)
        return prediction

    def softmax_tensor(self, input_tensor):
        """
        Softmax function that can be applied to a 
        three dimensional tensor.
        """
        d0, d1, d2 = input_tensor.shape
        reshaped = T.reshape(input_tensor, (d0*d1, d2))
        softmax_reshaped = T.nnet.softmax(reshaped)
        softmax = T.reshape(softmax_reshaped, newshape=input_tensor.shape)
        return softmax

    def output(self):
        output = self.activations['output_t'].get_value()
        return output

    def hidden(self):
        hidden = self.activations['hidden_t'].get_value()
        return hidden

