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
    def __init__(self, input_size, hidden_size, sigma_init, **kwargs):
        """
        The SRN is fully described by three weight matrices connecting
        the different layers of the network.

        :param input_size: number of input units
        :param hidden_size: number of hidden units
        :param embeddings: optional argument, if cotraining of word
        embeddings is required, one can give in an initial word embeddings
        matrix by passing an argument with the name embeddings
        :param classifier: optional argument, a softmax classifier is put
        on top of the network to map the input back to the correct size.
        One can provide initial values for the classifier by passing an argument
        with the name "classifier"
        """

        self.learning_rate = 0.5

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

        # weights to co-train embeddings

        embeddings_value = kwargs.get('embeddings', np.identity(input_size).astype(theano.config.floatX))
        self.embeddings = theano.shared(
                value = embeddings_value,
                name = 'embeddings'
        )

        # classifier on top to interpret output
        classifier_value = kwargs.get('classifier', np.random.uniform(-0.5, 0.5, (input_size, embeddings_value.shape[0])).astype(theano.config.floatX))
        self.classifier = theano.shared(
                value = classifier_value,
                name = 'classifier'
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        
    def generate_update_function(self):
        """
        Generate a symbolic expression describing how the network
        can be updated.
        """
        # current input and current hidden vector
        input_t = T.vector("input_t")

        # hidden_t = T.vector("hidden_t")   # TODO can we actually leave this out?

        input_map_t = T.dot(input_t, self.embeddings)
        hidden_next = T.nnet.sigmoid(T.dot(input_map_t, self.U) + T.dot(self.activations['hidden_t'], self.V) + self.b1)

        output_next = T.nnet.sigmoid(T.dot(self.activations['hidden_t'], self.W) + self.b2)
        prediction = self.prediction(self.activations['output_t'])

        updates = OrderedDict(zip(self.activations.values(), [hidden_next, output_next]))

        # givens = {}
        self.forward_pass = theano.function([input_t], updates=updates, givens={})
        self.cur_prediction = theano.function([], prediction)

        return

    def generate_network_dynamics(self, word_embeddings=False, classifier=True):
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

        # set the parameters of the network
        self.set_network_parameters(word_embeddings, classifier)
        self.print_embeddings = theano.function([], self.embeddings)

        # declare variables
        input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)
        input_sequences_transpose = input_sequences.transpose(1,0,2)

        # compute input map from input
        input_sequences_map = T.dot(input_sequences, self.embeddings)

        self.print_input_map = theano.function([input_sequences], input_sequences_map)

        # transpose to loop over right dimensions
        input_sequences_map_transpose = input_sequences_map.transpose(1,0,2)

        # describe how the hidden layer can be computed from the input
        def calc_hidden(input_map_t, hidden_t):
            return T.nnet.sigmoid(T.dot(input_map_t, self.U) + T.dot(hidden_t, self.V) + self.b1)

        hidden_t = T.matrix("hidden_t", dtype=theano.config.floatX)

        # compute sequence of hidden layer activations
        hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences_map_transpose, outputs_info=hidden_t)
        # compute prediction sequence (i.e., output layer activation)
        pre_output_sequences = T.nnet.sigmoid(T.dot(hidden_sequences, self.W) + self.b2)[-2]       # predictions for all but last output
        
        # compute softmax output
        # TODO should I have a third bias vector here?
        output_sequences = T.nnet.softmax(T.dot(pre_output_sequences, self.classifier))

        # compute predictions and target predictions
        predictions = T.argmax(output_sequences, axis = 1)
        target_predictions = T.argmax(input_sequences_transpose[-1], axis = 1)

        # compute sum squared differences between predicted numbers
        spe = T.sqr(predictions - target_predictions)   # squared prediction error per item
        sspe = T.sum(spe - target_predictions)           # sum squared prediction error

        # compute the difference between the output vectors and the target output vectors
        # errors = T.sqrt(T.sum(T.sqr(output_sequences - input_sequences_map_transpose[-1])))
        errors = T.nnet.categorical_crossentropy(output_sequences, input_sequences_transpose[-1])
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
        self.prediction_error_diff = theano.function([input_sequences], T.sqrt(spe), givens=givens)

        # take the sum of the latter for the whole batch
        self.prediction_err_diff_sum = theano.function([input_sequences], sspe, givens=givens)

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
            self.update_function(batch) 

        return

    def make_batches(self, input_sequences, batchsize):
        """
        Make batches from input sequence. 
        Currently this doesn't do anything but return the
        input sequence (for testing phase) but later this
        should start doing some more things.
        """
        # create indices for batches
        data_size = len(input_sequences)
        indices = np.random.permutation(data_size)

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

        return batches
    
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

    def set_network_parameters(self, word_embeddings, classifier):
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

