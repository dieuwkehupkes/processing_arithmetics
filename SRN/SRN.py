# A class representing a simple recurrent neural network

import numpy as np
import copy
from softmax import softmax


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
        Update all layers of the network
        """
        # set values inputLayer
        if input:
            self.I = input
        else:
            self.I = np.zeros(self.I.size())

        # make copy values hidden layer
        hiddenCopy = copy.deepcopy(self.H)

        # set values hiddenLayer
        self.H = np.tanh(self.I.dot(self.Wih) + self.C.dot(self.Wch))

        # set values outputLayer
        self.O = softmax(hiddenCopy.dot(self.Who))

        # set values contextLayer
        self.contextLayer = hiddenCopy

    def train(self, input_output, rounds=1):
        """
        Train the network with backpropagation to always
        predict the next vector in the inputSequence
        :param input_output: an array with tuples of input-output pairs
        """
        for round in xrange(rounds):
            self.reset()
            self.I = input_output[0]
            self.update()
            for pair in xrange(1, input_output.size()):
                raise NotImplementedError

    def reset(self):
        """
        Reset the values of all layers.
        """
        self.I = np.zeros(self.I.size)
        self.H = np.zeros(self.H.size)
        self.C = np.zeros(self.C.size)
        self.O = np.zeros(self.O.size)
