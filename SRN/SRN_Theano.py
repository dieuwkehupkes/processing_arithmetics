"""
Dieuwke Hupkes - <dieuwkehupkes@gmail.com>
"""

import numpy as np
import theano
import theano.tensor as T


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

        # create variables for hidden layers
        # TODO do we even need this?
        self.input_t = T.vector("input_t", dtype=theano.config.floatX)
        self.hidden_t = T.vector("hidden_t", dtype=theano.config.floatX)
        self.output_t = T.vector("output_t", dtype=theano.config.floatX)

        # store dimensions of network
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # store the parameters of the network
        self.params = [self.U, self.V, self.W]
        # @TODO check if this is also updated when the parameters are updated

    def generate_update_function(self):
        """
        Generate a symbolic expression describing how the network
        can be updated.
        """
        # current input
        x = T.vector("input")

        # TODO generate update function!!!
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
        # TODO something is off with this input sequence, that has to be set as an
        #      attribute of the network now, change that so that it makes more sence

        # function describing how the hidden layer can be computed from the input
        input_sequence = T.matrix("input_sequence", dtype=theano.config.floatX)
        def calc_hidden(input_t, hidden_t):
            return T.nnet.sigmoid(T.dot(input_t, self.U) + T.dot(hidden_t, self.V))

        hidden_t = T.vector("hidden_t", dtype=theano.config.floatX)

        # compute sequence of hidden layer activations
        hidden_sequence, _ = theano.scan(calc_hidden, sequences=input_sequence, outputs_info = hidden_t)


        # compute prediction sequence (i.e., output layer activation)
        output_sequence = T.nnet.softmax(T.dot(hidden_sequence, self.W))[:-1]       # predictions for all but last output

        # symbolic definition of error
        errors = T.nnet.categorical_crossentropy(output_sequence, input_sequence[1:])
        error = T.mean(errors)

        # gradients
        gradients = T.grad(error, self.params)

        # updates
        new_params = [self.params[i] - self.learning_rate*gradients[i] for i in xrange(len(self.params))]


        # inputs
        # @TODO should I put this somewhere else?
        # TODO klopt dit??? of moet ik echte waardes meegeven
        givens = {
                hidden_t:       np.zeros(self.hidden_size).astype(theano.config.floatX),
                # input_t:        np.zeros(self.input_size).astype(theano.config.floatX)
        }

        self.update_function = theano.function([input_sequence], new_params, givens=givens)        # @TODO klopt updates=+, 

        return

    def train(self, input_sequences, no_iterations, some_other_params):
        """
        Train the network to store input_sequences
        :param input_sequences  
        :param no_iterations    
        """
        #TODO write function description
        for iteration in xrange(0, no_iterations):
            iteration(input_sequences)

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
        return input_sequences

    def prediction(self):

        # assuming a 1-k encoding, the network prediction is the output unit
        # with the highest activation value after applying the sigmoid
        # If we are using distributed representations oid (what in the end
        # possibly desirable is, the symbolic expression for the prediction
        # for the prediction would change (as well as the output activation, btw)
        self.prediction = T.argmax(self.output, axis=1)

