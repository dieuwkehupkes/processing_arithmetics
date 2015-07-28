# A class representing a simple recurrent neural network
# The setup is not very re-usable, if I want to do more things
# with this later I'll take the code and make sure the training algorithm
# and these things are a little more extendabel

import numpy as np
import copy
from softmax import *
from sigmoid import *


class SimpleRecurrentNetwork():
    """
    A class representing a simple recurrent network (SRN), as presented
    in Elman (1990).
    """
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
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
        self.Wch = np.zeros((hiddenLayer, hiddenLayer))
        self.Who = np.zeros((hiddenLayer, outputLayer))

    def set_input_to_hidden(self, weight_matrix):
        """
        Set weights from input to hidden layer to weight_matrix
        """
        if weight_matrix.size != self.Wih.size:
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Wih = weight_matrix

    def set_hidden_to_output(self, weight_matrix):
        """
        Set weights from hidden to output layer to weight_matrix
        """
        if weight_matrix.size != self.Who.size:
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Who = weight_matrix

    def set_context_to_hidden(self, weight_matrix):
        """
        Set weights from context to hidden layer to weight_matrix
        """
        if weight_matrix.size != self.Wch.size:
            print("Incorrect inputsize, weightMatrix not changed")
        else:
            self.Wch = weight_matrix

    def initialise_weights(self, min_value, max_value):
        """
        Initialise weights randomly between minvalue
        and maxvalue.
        """
        factor = max_value - min_value
        shift = min_value
        self.Wih = factor * np.random.rand(self.Wih.shape[0], self.Wih.shape[1]) + shift
        self.Wch = factor * np.random.rand(self.Wch.shape[0], self.Wch.shape[1]) + shift
        self.Who = factor * np.random.rand(self.Who.shape[0], self.Who.shape[1]) + shift

    def update(self, input=None):
        """
        Update all layers of the network until input
        has reached the end (i.e., the output and hidden layers
        are updated multiple times within one "update")
        @TODO: check if .dot is indeed the way to multiply
        """
        # set values inputLayer
        if input != None:
            self.I = input
            if self.I.size != input.size:
                raise ValueError("Input size doesn't match size of input layer")
        else:
            self.I = np.zeros(self.I.size)

        # set values hiddenLayer
        self.H = sigmoid(self.I.dot(self.Wih) + self.C.dot(self.Wch))

        # set values outputLayer
        # self.O = sigmoid(self.H.dot(self.Who))
        self.O = softmax(self.H.dot(self.Who))

        # set values contextLayer to hidden layer
        self.C = self.H

        # print "input layer: ", self.I
        # print "hidden layer: ", self.H
        # print "output layer: ", self.O
        # exit()


    def train(self, sequences, learning_rate = 0.1, rounds=1, depth=1):
        """
        Train the network with backpropagation to always
        predict the next vector in the inputSequence
        :param input_output: an array with tuples of input-output pairs
        """

        for round in xrange(rounds):

            # print "round ", round
            self.reset()

            # create new training sequence by shuffling and then unpacking
            training_sequence = [input_state for sequence in sequences for input_state in sequence]
            
            # create arrays to store previous states
            prev_hidden = np.zeros((depth, self.H.size))
            index_hidden = 0

            # loop through training sequence
            for index in xrange(len(training_sequence)-1):
                    training_example = training_sequence[index]

                    self.update(training_example)
                    prev_hidden[index_hidden] = self.H              # store hidden state for BPTT

                    # Compute error signal output
                    diff_target = training_sequence[index+1] - self.O    # compute output error
                    # print "output activation:", self.O
                    # print "target: ", training_sequence[index+1], 'output: ', self.O
                    # jacobian = jacobian_sigmoid(self.O)
                    jacobian = jacobian_softmax(self.O)                             # compute jacobian matrix with partial derivatives
                    # print "jacobian", jacobian
                    error_signal = np.dot(diff_target, jacobian)

                    update_Who = learning_rate * np.outer(self.H, error_signal)
                    # print "hidden layer activation: ", self.H
                    # print "error signal", error_signal
                    # print "update hidden to output:", update_Who
                    # raw_input()

                    # set working timelag
                    time_lag = 0

                    # initialise update matrices for Wch and Wih
                    update_Wch = np.zeros(self.Wch.shape)
                    update_Wih = np.zeros(self.Wih.shape)

                    # compute first error signal for hidden layer
                    jacobian = jacobian_sigmoid(self.H)
                    error_signal = np.dot(np.dot(self.Who, error_signal), jacobian)

                    while time_lag <= depth and index >= time_lag:

                        #print "time_lag:", time_lag

                        # update Wch & Wih
                        update_Wch += learning_rate * np.outer(prev_hidden[index_hidden-time_lag], error_signal)

                        # update Wih
                        update_Wih += learning_rate * np.outer(self.I, error_signal)

                        # propagate error one time step back
                        error_signal = np.dot(np.dot(self.Wch, error_signal), jacobian)
                        time_lag += 1

                    # update weights
                    self.Wch += update_Wch
                    self.Who += update_Who
                    self.Wih += update_Wih

                    # print "update weights input to hidden: ", update_Wih
                    # print "update weights context to hidden", update_Wch
                    # print "\n\n"
                    # raw_input()


                    index_hidden = (index_hidden + 1) % depth       # switch index of storing hidden state


    def reset(self):
        """
        Reset the values of all layers.
        """
        self.I = np.zeros(self.I.size)
        self.H = np.zeros(self.H.size)
        self.C = np.zeros(self.C.size)
        self.O = np.zeros(self.O.size)
