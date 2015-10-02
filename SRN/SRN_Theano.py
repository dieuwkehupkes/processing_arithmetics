"""
Dieuwke Hupkes - <dieuwkehupkes@gmail.com>
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
    def __init__(self, input_size, hidden_size, output_size, sigma_init):
        """
        The SRN is fully described by three weight matrices connecting
        the different layers of the network.

        :param input_size: number of input units
        :param hidden_size: number of hidden units
        :param output_size: number of output units
        """
        # TODO as size(input) == size(output) we don't need to
        # take that as a parameter really, or otherwise
        # build in a check in the training method

        self.learning_rate = 0.1

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
                    (hidden_size, output_size)
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
                    0, sigma_init, output_size
                ).astype(theano.config.floatX),
                name = 'b2'
        )

        # Store the network activation values to run the network
        hidden_t = theano.shared(
                value = np.zeros(hidden_size).astype(theano.config.floatX),
                name = 'hidden_t'
        )
        output_t = theano.shared(
                value = np.zeros(output_size).astype(theano.config.floatX),
                name = 'output_t'
        )

        self.activations = OrderedDict(zip(['hidden_t','output_t'], [hidden_t, output_t]))

        # store dimensions of network
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
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
        # current input
        input_t = T.vector("input_t")
        hidden_t = T.vector("hidden_t")

        hidden_next = T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(self.activations['hidden_t'], self.V) + self.b1)

        output_next = T.flatten(T.nnet.softmax(T.dot(self.activations['hidden_t'], self.W) + self.b2))        # output_next = T.nnet.softmax(T.dot(self.activations['hidden_t'], self.W))

        updates = OrderedDict(zip(self.activations.values(), [hidden_next, output_next]))
        # updates = OrderedDict([(self.activations['hidden_t'], hidden_next)])

        # givens = {}
        self.forward_pass = theano.function([input_t], updates=updates, givens={})

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

        # TODO Maybe I should organise this differently somehow
        # TODO it would be nice if I could also compute the prediction error
        #      attribute of the network now, change that so that it makes more sence

        # function describing how the hidden layer can be computed from the input
        input_sequence = T.matrix("input_sequence", dtype=theano.config.floatX)
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
        prediction_error = 1 - T.mean(T.eq(predictions, self.prediction(input_sequence[1:])))   # scalar

        # gradients
        gradients = OrderedDict(zip(self.params.keys(), T.grad(error, self.params.values())))

        # updates for weightmatrices and historical grads
        new_params = []
        for param in self.params:
            new_histgrad = self.histgrad[param] + T.sqr(gradients[param])
            new_param_value = self.params[param] - gradients[param]/(T.sqrt(new_histgrad) + 0.000001)
            new_params.append((self.params[param], new_param_value))
            new_params.append((self.histgrad[param], new_histgrad))

        # updates for weight matrices
        # new_params_values = [self.params[param] - self.learning_rate*gradients[param] for param in self.params.keys()]
        # new_params = zip(self.params.values(), new_params_values)

        # initial values
        givens = {
                hidden_t:       np.zeros(self.hidden_size).astype(theano.config.floatX),
        }

        self.update_function = theano.function([input_sequence], updates=new_params, givens=givens)        
        self.compute_error = theano.function([input_sequence], error, givens=givens)

        self.compute_prediction_error = theano.function([input_sequence], prediction_error, givens=givens)

        return

    def train(self, input_sequences, no_iterations, some_other_params=None):
        """
        Train the network to store input_sequences
        :param input_sequences  
        :param no_iterations    
        """
        #TODO write function description
        for iteration in xrange(0, no_iterations):
            self.iteration(input_sequences)

        return

    def iteration(self, input_sequences):
        """
        Slice data in minibatches and perform one
        training iteration.
        :param input_sequences: The sequences we want to
                                store in the network
        """
        batches = self.make_batches(input_sequences)

        # loop over minibatches, update parameters
        for batch in batches:
            self.update_function(batch)

        return

    def make_batches(self, input_sequences):
        """
        Make batches from input sequence. 
        Currently this doesn't do anything but return the
        input sequence (for testing phase) but later this
        should start doing some more things.
        """
        # TODO Make that this method actually does something
        # return permutated version of input sequences
        return np.random.permutation(input_sequences)

    def prediction(self, output_vector):

        # assuming a 1-k encoding, the network prediction is the output unit
        # with the highest activation value after applying the sigmoid
        # If we are using distributed representations oid (what in the end
        # possibly desirable is, the symbolic expression for the prediction
        # for the prediction would change (as well as the output activation, btw)
        prediction = T.argmax(output_vector, axis=1)
        return prediction

    def output(self):
        output = self.activations['output_t'].get_value()
        return output

    def hidden(self):
        hidden = self.activations['hidden_t'].get_value()
        return hidden

