"""
Base class for RNN's.
Dieuwke Hupkes - <D.hupkes@uva.nl>
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import itertools


class RNN():
    """
    A class providing basic functionality for RNN's.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, sigma_init=0.2, **kwargs):
        """
        This class provides functionality for RNNs in Theano.

        A basic RNN consists of an input, a hidden and and output layer.
        The behaviour of the RNN is fully described by three weight matrices
        and a set of activation functions:

            X_t = f(V*X_{t-1} + U*I_t + b1)
            Y_t = g(W*X_t + b2)

        Where X_t is the hidden layer, Y_t is the output layer and f and g
        are activation functions describing how to compute the current
        activations of the layers given their netto input.

        Args:
        :param input_size:      number of input units
        :param hidden_size:     number of hidden units
        :param output_size:     number of output units
        :param sigma_init:      parameter for initialising
                                weight matrices
        :param learning_rate:   learning rate for training

        Kwargs:
        :param embeddings:  if cotraining of word embeddings is desired
                            one can give in an initial word embeddings
                            matrix by passing passing an argument with
                            the name embeddings
        :param classifier:  a softmax classifier is put on top of the
                            network to map the input back to the correct
                            size. One can provide initial values for
                            the classifier by passing an argument
                            with the name "classifier"
        """

        self.learning_rate = learning_rate

        # weights from input to hidden
        self.U = theano.shared(
                value = sigma_init*np.random.normal(
                    input_size, hidden_size
                ),
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

        # bias for hidden layer
        self.b1 = theano.shared(
                value = np.random.normal(
                    0, sigma_init, hidden_size
                ).astype(theano.config.floatX),
                name='b1'
        )

        # bias for output layer
        self.b2 = theano.shared(
                value = np.random.normal(
                    0, sigma_init, input_size
                ).astype(theano.config.floatX),
                name = 'b2'
        )

        # weights to co-train embeddings
        embeddings_value = kwargs.get('embeddings', np.identity(input_size).astype(theano.config.floatX))
        self.embeddings = theano.shared(
                value = embeddings_value,
                name = 'embeddings'
        )

        # Store the network activation values to run the network
        # TODO Maybe I don't need this, check later if it is still useful
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        
    def generate_forward_pass(self):
        """
        Symbolically define how a batch of output sequences
        can be computed from a batch of input sequences.

        This function depends on the specific details of the
        network and should be defined in the subclass.
        """
        raise NotImplementedError("Function not implemented in abstract class")

    def generate_standard_training_function(self):
        """
        Generate a function that allows standard training
        of the network with batches and backpropagation
        """
        # TODO make this description more elaborate

        
        # set trainable parameters network

        # generate symbolic input variables

        # generate output sequences for input using forward function (symbolic)

        # compare with target outputs, generate error signal

        # compute gradients

        # compute weight updates and store in ordered dictionary
        
        # make givens dictionary

        self.train = self.training_step_standard([input_sequences], updates=updates, givens=givens)

        raise NotImplementedError("implement function generate_standard_training_function")

        return


    def generate_comparison_training1_function(self):

        # generate comparison matrix

        # generate softmax classifier on top
        
        # set trainable the trainable parameters for this configuration
        # (in this case: all weight matrices, softmax classifier and comparison matrix)

        # generate symbolic variables for input of training function
        
        # generate output for batch

        # apply comparison matrix

        # use softmax layer to compute outputs

        # compare with target outputs using cross_entropy

        # compute gradients for all trainable parameters

        # store weight updates in ordered dictionary

        # generate givens dictionary
        
        # generate update function
        self.training_step_comparison1 = theano.function([batch1, target], updates=updates, givens=givens)

        raise NotImplementedError("Implement this function")

        return

    def train(self, input_sequences, no_iterations, batchsize, some_other_params=None):
        """
        Train the network to store input_sequences
        :param input_sequences  
        :param no_iterations    
        """
        # TODO write function description
        for iteration in xrange(0, no_iterations):
            self.iteration(input_sequences, batchsize)

        return

    def comparison_training1(self, input_sequences1, target_sequence, no_iterations, batch_size):
        """
        Train the network by comparing the output of an
        input sequence with a random number. Use a comparison
        layer and a softmax classifier on top to propagate
        back error signal.
        """
        # TODO check if input sequences and targets have same length
        raise NotImplementedError("Implement check of length")
        
        # iterate for no_iterations steps
        for iteration in xrange(0, no_iterations):
            # create new batches
            batches, indices = self.make_batches(input_sequences1, batch_size)
            targets, _ = self.make_batches(target_sequence, indices)

        for batch1, target in itertools.izip(batches, targets):
            # update weights for current batch
            self.training_step_comparison1(batch1, targets)

        return

    def comparison_training2(self, input_sequences1, input_sequences2, target_sequence, no_iterations, batch_size):
        """
        Train the network by comparing the output of two
        input sequences using a comparison layer and a
        softmax classifier on top.
        """

        # TODO check if input sequences have equal number of sequences
        raise NotImplementedError("Implement check to see if sequences have same length")
        
        # iterate for no_iterations steps
        for iteration in xrange(0, no_iterations):
            # create new batches
            batches1, indices = self.make_batches(input_sequences1, batch_size)
            batches2, _ = self.make_batches(input_sequences2, batch_size, indices=indices)
            targets, _ = self.make_batches(target_sequence, batch_size, indices=indices)

            for batch1, batch2, target in itertools.izip(batches1, batches2, targets):
                # update weights for current batch
                self.training_step_comparison2(batch1, batch2, targets)

        return

    def iteration(self, input_sequences, batchsize):
        """
        Slice data in minibatches and perform one
        training iteration.
        :param input_sequences: The sequences we want to
                                store in the network
        """
        batches, _ = self.make_batches(input_sequences, batchsize)

        # loop over minibatches, update parameters
        for batch in batches:
            self.training_step_standard(batch) 

        return

    def make_batches(self, input_sequences, batchsize, **kwargs):
        """
        Make batches from input sequence. 
        """
        # create indices for batches
        data_size = len(input_sequences)
        indices = kwargs.get(indices, np.random.permutation(data_size))

        # create array for batches
        batches = []
        to, fro = 0, batchsize
        while fro <= data_size:
            batch = input_sequences[indices[to:fro]]
            batches.append(batch)
            to = fro
            fro += batchsize

        # last batch
        batch = input_sequences[to:]
        batches.append(batch)

        return batches, indices
    
    def prediction(self, output_vector):
        """
        Compute the prediction of the network for output_vector
        by comparing its output with the word embeddings matrix.
        The prediction of the network will be the word embedding
        the output vector is closest to.
        NB: this function only works for an input *vector* and
        cannot be applied to an input matrix of vectors
        """
        raise NotImplementedError("Implement function in subclass")
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

    def set_network_parameters(self, word_embeddings):
        """
        Set trainable parameters of the network.
        """
        # store the parameters of the network
        self.params = OrderedDict([('U', self.U), ('V', self.V), ('W', self.W), ('b1', self.b1), ('b2', self.b2)])

        if word_embeddings:
            self.params['embeddings'] = self.embeddings

        return

    def set_histgrad(self):
        """
        Initialise historical grad for training with
        adagrad.
        """

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

