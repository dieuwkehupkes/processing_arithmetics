"""
Dieuwke Hupkes - <dieuwkehupkes@gmail.com>
"""

import numpy as np
import theano
import theano.tensor as T


class SimpleRecurrentNetwork():
    """
    A class representing a simple recurrent network (SRN), as presented
    in Elman (1990).
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        The SRN is fully described by three weight matrices connecting
        the different layers of the network.

        :param input_size: number of input units
        :param hidden_size: number of hidden units
        :param output_size: number of output units
        """

        # @TODO init random ipv zeros, gebruik rng random streams

        # weights from input to hidden
        self.U = theano.shared(value=numpy.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='U')

        # weights from context to hidden
        self.V = theano.shared(value=numpy.zeros((hidden_size, hidden_size), dtype=theano.config.floatX), name='V')

        # weights from hidden to output
        self.W = theano.shared(value=numpy.zeros((hidden_size, output_size), dtype=theano.config.floatX), name='W')
        
        # store the parameters of the network
        self.params = [self.U, self.V, self.W]
        # @TODO check if this is also updated when the parameters are updated

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

        # function describing how the hidden layer can be computed from the input
        self.input_sequence = T.matrix("input_sequence")
        def hidden(input_t, hidden_t):
            return T.nnet.sigmoid(T.dot(self.input_t, self.U) + T.dot(self.hidden_t, self.V))

        # compute sequence of hidden layer activations
        self.hidden_t = T.vector("hidden")
        hidden_sequence, _ = theano.scan(hidden, sequence=input_sequence, outputs_info = self.hidden_t)

        # compute prediction sequence (i.e., output layer activation)
        output_sequence = T.nnet.softmax(T.dot(hidden_sequence, self.W))[:-1]       # predictions for all but last output

        # symbolic definition of error
        errors = categorical_crossentropy(output_sequence - input_sequence[1:])
        error = T.mean(errors)

        # gradients
        gradients = T.grad(error, self.params)

        # updates @TODO is the sign of the update correct?
        new_params = self.params - gradients*learning_rate

        # inputs
        # @TODO this should be somewhere else I think
        givens = {
                input_sequence = data
                h_t = h_value           # @TODO give expression for h_t, potentially have a value, initialise h
        }

        self.update_function = theano.function(input_data, error, updates=new_params, givens=givens)        # @TODO klopt updates=+, 


        return

    def prediction(self):

        # assuming a 1-k encoding, the network prediction is the output unit
        # with the highest activation value after applying the sigmoid
        # If we are using distributed representations oid (what in the end
        # possibly desirable is, the symbolic expression for the prediction
        # for the prediction would change (as well as the output activation, btw)
        self.prediction = T.argmax(self.output, axis=1)



"""
Otto:
volgens mij moeten we nu eerst voorspelling uitrekenen, dan je error berekenen, dan gradients definieren als afgeleide van error wrt je parameters, dan de learning update definieren, en dan de gegeven (givens) definieren en functie aanmaken:) dus eerst prediction uitrekenen
en dan  learning update definieren, en dan de gegeven (givens) definieren
en functie aanmaken:)
"""


