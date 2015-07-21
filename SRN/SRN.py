# A class representing a simple recurrent neural network
# The setup is not very re-usable, if I want to do more things
# with this later I'll take the code and make sure the training algorithm
# and these things are a little more extendabel

import numpy as np
import copy
import itertools
from softmax import all
from sigmoid import all


class SimpleRecurrentNetwork():
    """
    A class representing a simple recurrent network (SRN), as presented
    in Elman (1990).
    """
    def __init(self, inputLayer, hiddenLayer, outputLayer):
        """
        :param inputLayer: size of the input layer
        :type inputLayer: int
        :param hiddenLayer size of hidden layer
        :type hiddenLayer: int
        :param outputLayer: size of output layer
        """
        self.I = np.zeros(inputLayer)       # matrix to hold input layer values
        self.H = np.zeros(hiddenLayer)      # matrix to hold hidden layer values
        self.C = np.zeros(hiddenLayer)      # matrix to hold context layer values
        self.O = np.zeros(outputLayer)       # matrix to hold output layer values

        # initialise weights
        self.Wih = np.zeros((inputLayer, hiddenLayer))
        self.Wch = np.zeros((inputLayer, hiddenLayer))
        self.Who = np.zeros((hiddenLayer, outputLayer))

    def set_input_to_hidden(self, weight_matrix):
        """
        Set weights from input to hidden layer to weight_matrix
        """
        if weight_matrix.size() != self.Wih.size():
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Wih = weight_matrix

    def set_hidden_to_output(self, weight_matrix):
        """
        Set weights from hidden to output layer to weight_matrix
        """
        if weight_matrix.size() != self.Who.size():
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Who = weight_matrix

    def set_context_to_hidden(self, weight_matrix):
        """
        Set weights from context to hidden layer to weight_matrix
        """
        if weight_matrix.size() != self.Wch.size():
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Wch = weight_matrix

    def update(self, input=None):
        """
        Update all layers of the network until input
        has reached the end (i.e., the output and hidden layers
        are updated multiple times within one "update")
        @TODO: check if .dot is indeed the way to multiply
        """
        # set values inputLayer
        if input:
            self.I = input
        else:
            self.I = np.zeros(self.I.size())

        # set values hiddenLayer
        self.H = np.tanh(self.Wih.dot(self.I) + self.Wch.dot(self.C))

        # set values outputLayer
        self.O = softmax(self.Who.dot(self.H))

        # set values contextLayer to hidden layer
        self.C = self.H

    def train(self, sequences, learning_rate = 0.1, rounds=1, depth=1):
        """
        Train the network with backpropagation to always
        predict the next vector in the inputSequence
        :param input_output: an array with tuples of input-output pairs
        """

        for round in xrange(rounds):
            self.reset()

            # create new training sequence by shuffling and then unpacking
            training_sequence = list(itertools.chain(np.random.permutation(sequences)))
            
            self.I = training_sequence[0]
            self.update()

            # create arrays to store previous states
            prev_hidden = np.zeros((depth, self.H.size))
            index_hidden = 0

            # loop over training examples
            for index in xrange(len(training_sequence)-1):
                    training_example = training_sequence[index]

                    self.I = training_example
                    self.update()
                    prev_hidden[index_hidden] = self.H              # store hidden state

                    # update weights Who
                    output_error = training_sequence[index+1] - training_example    # compute output error
                    jacobian = jacobian_softmax(self.O)                             # compute jacobian matrix with partial derivatives
                    update_Who = learning_rate * jacobian * self.H * output_error     # compute update (check even of dit goed gaat met de assen e.d.)

                    # propagate error back to H
                    hidden_error = self.O.dot(jacobian)                             # hier gaat ongetwijfeld ook iets mis, check dit even

                    # set working timelag
                    time_lag = 0
                    update_Wch = np.zeros(self.Wch.shape)

                    while time_lag <= depth and index >= time_lag:

                        # update Wch
                        jacobian = jacobian_sigmoid(self.H)
                        update_Wch += learning_rate * jacobian * prev_hidden[index_hidden-time_lag] * hidden_error      # compute update for Wch

                        # update Wih
                        update_Wch += learning_rate * jacobian * training_sequence[index-time_lag] * [index_hidden-time_lag] * hidden_error      # compute update for Wch

                        # propagate error one time step back
                        hidden_error = prev_hidden[index].dot(jacobian)

                        time_lag += 1

            index_hidden = (index_hidden + 1) % depth       # switch index of storing hidden state

    def reset(self):
        """
        Reset the values of all layers.
        """
        self.I = np.zeros(self.I.size)
        self.H = np.zeros(self.H.size)
        self.C = np.zeros(self.C.size)
        self.O = np.zeros(self.O.size)
