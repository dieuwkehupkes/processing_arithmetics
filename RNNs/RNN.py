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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, sigma_init=0.2, embeddings=False, **kwargs):
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
        :param embeddings:      cotrain word embeddinsg if set to true

        Kwargs:
        :param embeddings_init: one can give in an initial word embeddings
                                matrix by passing passing an argument with
                                the name embeddings
        :param classifier:      a softmax classifier is put on top of the
                                network to map the input back to the correct
                                size. One can provide initial values for
                                the classifier by passing an argument
                                with the name "classifier"
        """

        self.learning_rate = learning_rate
        self.train_embeddings = embeddings

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
                    (hidden_size, output_size)
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
                    0, sigma_init, output_size
                ).astype(theano.config.floatX),
                name = 'b2'
        )

        # weights for word embeddings
        embeddings_value = kwargs.get('embeddings_init', np.identity(input_size).astype(theano.config.floatX))
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
                value = np.zeros(output_size).astype(theano.config.floatX),
                name = 'output_t'
        )

        self.activations = OrderedDict(zip(['hidden_t','output_t'], [hidden_t, output_t]))

        # store dimensions of network
        self.embeddings_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = embeddings_value.shape[1]
        
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
        self.set_network_parameters(self.train_embeddinsg)
        self.set_histgrad()

        # generate symbolic input variables
        batch = T.tensor3("batch", dtype=theano.config.floatX)
        targets = T.matrix("targets", dtype=theano.config.floatX)

        # generate output sequences for input using forward function (symbolic)
        output_sequences = self.return_outputs(batch)[-1]

        # compute mean squared error
        error = T.sum(T.sqr(output_sequences - targets))

        # compute gradients
        grads = T.grad(error, self.params.values())
        gradients = OrderedDict(zip(self.params.keys(), grads))

        # compute weight updates and store in ordered dictionary
        updates = self.compute_weight_updates(gradients)

        # generate update function
        self.training_step_standard = theano.function([batch, targets], updates=updates, givens={})

        return

    def generate_comparison_training1_function(self):
        """
        Describe what this function does.
        """
        # set trainable the trainable parameters for this configuration
        self.set_network_parameters(self.train_embeddings)
        self.add_training_framework()
        self.add_trainable_parameters(('comparison', self.comparison), ('classifier', self.classifier), ('b3', self.b3), ('b4', self.b4))
        self.set_histgrad()

        # generate symbolic variables for input of training function
        batch = T.tensor3("batch", dtype=theano.config.floatX)  # tensor 3dim
        comparison_sequences = T.matrix("comparison_sequences", dtype=theano.config.floatX)  # tensor 2-dim
        targets = T.vector("targets", dtype=theano.config.floatX)
        
        # generate output for batch, take last slice to get final representations
        output_sequences = self.return_outputs(batch)[-1]
        
        # apply comparison matrix
        comparison_layer = T.nnet.sigmoid(T.dot(output_sequences, self.comparison) + self.b3)

        # use softmax layer to compute outputs
        outputs = T.nnet.softmax(T.dot(comparison_layer, self.classifier) + self.b4)

        # compare with target outputs using cross_entropy
        error = T.mean(T.nnet.categorical_crossentropy(outputs, targets))

        # compute gradients for all trainable parameters
        grads = T.grad(error, self.params.values())
        gradients = OrderedDict(zip(self.params.keys(), grads))

        # compute weight updates
        updates = self.compute_weight_updates(gradients)

        # generate update function
        self.training_step_comparison1 = theano.function([batch, comparison_sequences, targets], updates=updates, givens={})

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

    def add_trainable_parameters(self, *args):
        """
        Add arguments in arg to trainable parameters
        of the network.
        :param args:    a tuple with a name (string) and a
                        shared variable that is set as attribute
                        of the network
        """
        for name, var in args:
            self.params[name] = var

        return

    def add_training_framework(self):
        """
        Add comparison layer and classifier and
        corresponding biases to parameters of 
        network to do comparison training.
        """

        # following Bowman, set size comparison layer
        # to 3 times hidden_size
        size_comparison_layer = self.hidden_size

        # generate comparison matrix
        self.comparison = theano.shared(
                value = self.sigma_init*np.random.normal(
                    2*self.output_size, size_comparison_layer
                ).astype(theano.config.floatX),
                name='comparison'
        )

        # classifier to compare two inputs
        self.classifier = theano.shared(
                value = self.sigma_init*np.random.normal(
                    size_comparison_layer, 3
                ).astype(theano.config.floatX),
                name='classifier'
        )

        # biasses for classifier and comparison layer
        self.b3 = theano.shared(
                value = np.random.normal(
                    0, self.sigma_init, size_comparison_layer
                ).astype(theano.config.floatX),
                name='b3'
        )

        self.b4 = theano.shared(
                value = np.random.normal(
                    0, self.sigma_init, size_comparison_layer
                ).astype(theano.config.floatX),
                name='b4'
        )

        return

    def compute_weight_updates(self, gradients):
        """
        Compute the weight updates given the gradients,
        using the adagrad parameters stored as attributes
        of the network.
        Return an ordered dictionary with updates for
        all parameters.
        """
        new_params = OrderedDict()
        for param in self.params:
            new_histgrad = self.histgrad[param] + T.sqr(gradients[param])
            new_param_value = self.params[param] - self.learning_rate*gradients[param]/(T.sqrt(new_histgrad) + 0.000001)
            new_params[self.params[param]] = new_param_value
            new_params[self.histgrad[param]] = new_histgrad

        return new_params


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

