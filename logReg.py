# class to do logistic regression with Theano, for practising
import theano as T
import theano.Tensor als tensor
import numpy as np


class logReg(object):

    def __init__(self, input, n_in, n_out):
        # weight matrix
        self.W = T.shared(value=np.zeros((n_in, n_out), dtype=T.config.floatX), name='W', borrow=True)

        # biases
        self.b = T.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # output vector
        self.prob_output = tensor.nnet.softmax(T.dot(input, self.W) + self.b)

        # prediction (= argmax output vector)
        self.prediction = tensor.argmax(self.prob_output, axis=1)

        # parameters of model
        self.params = [self.W, self.b]


