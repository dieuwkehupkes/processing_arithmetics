import unittest
from keras.layers import GRU
from architectures import A1
from architectures import A4
import numpy as np
from generate_training_data import generate_dmap

class TestArchitectures(unittest.TestCase):
    """
    There are undoubtedly many things I should test here,
    but for now I am not really sure which ones and how
    to set it up.
    """

    def setUp(self):
        # check if training o
        pass
    

class TestTrain(unittest.TestCase):
    """
    Test if training still works.
    """

    def setUp(self):
        digits                      = np.arange(-10, 11)
        l_val             = {'L1':1, 'L2': 1}
        l_train             = {'L3':1}
        dmap, _, _ = generate_dmap(digits, l_train, l_val)
        self.dmap = dmap
        self.input_dim = len(dmap)+1
        maxlen = 20

        # create architectures
        self.A1 = A1()
        self.A4 = A4()

        # generate training data
        self.X_A1, self.Y_A1 = self.A1.generate_training_data(l_train, dmap=dmap, digits=digits, pad_to=maxlen)
        self.X_A4, self.Y_A4 = self.A4.generate_training_data(l_train, dmap=dmap, digits=digits, pad_to=maxlen)

        # generate validation data
        self.X_val_A1, self.Y_val_A1 = self.A1.generate_training_data(l_val, dmap=dmap, digits=digits, pad_to=maxlen)
        self.X_val_A4, self.Y_val_A4 = self.A4.generate_training_data(l_val, dmap=dmap, digits=digits, pad_to=maxlen)

        # generate models
        self.test_generate_model_A1()
        self.test_generate_model_A4()

    def test_generate_model_A1(self):
        # test if model can be generated, set it as attribute to class
        self.A1.generate_model(GRU, self.input_dim, input_size=2, input_length=20, size_hidden=2, dmap=self.dmap)

    def test_generate_model_A4(self):
        # test if model can be generated, set it as attribute to class
        self.A4.generate_model(GRU, input_dim=self.input_dim, input_size=2, input_length=20, size_hidden=2, dmap=self.dmap)

    def test_training_A1(self):
        # test training with part of training data as validation data
        self.A1.train(training_data=(self.X_A1, self.Y_A1), validation_data=None, validation_split=0.1, batch_size=2, epochs=1, verbosity=0)
 
    def test_training_A1_validation_data(self):
        # test training with separate validation data
        self.A1.train(training_data=(self.X_A1, self.Y_A1), validation_data=(self.X_val_A1, self.Y_val_A1), batch_size=2, epochs=1, verbosity=0)

    def test_training_A4(self):
        # test training with part of training data as validation data
        self.A4.train(training_data=(self.X_A4, self.Y_A4), validation_data=None, validation_split=0.1, batch_size=2, epochs=1, verbosity=0)

    def test_training_A4_validation_data(self):
        # test training with separate validation data
        self.A4.train(training_data=(self.X_A4, self.Y_A4), validation_data=(self.X_val_A4, self.Y_val_A4), batch_size=2, epochs=1, verbosity=0)


if __name__ == '__main__':
        unittest.main()
