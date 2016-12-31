import pytest
import numpy as np
import sys
sys.path.insert(0, '../arithmetics') 
from arithmetics import mathTreebank
from architectures import A1, A4, Probing, Seq2Seq, Training
from keras.layers import GRU
import pickle
import os

def test_A1_methods():
    _test_architecture_methods(A1)

def test_A4_methods():
    _test_architecture_methods(A4)

def test_Probing_methods():
    _test_architecture_methods(Probing, extra_classifiers=['subtracting'])

def test_Seq2Seq_methods():
    _test_architecture_methods(Seq2Seq)

def _test_architecture_methods(architecture, extra_classifiers=None):
    """
    Test if methods of Training class still work.
    """

    # generate architecture, load dmap
    A = architecture()
    dmap = pickle.load(open('best_models/dmap','r'))

    # test build model
    A.generate_model(recurrent_layer=GRU, input_dim=len(dmap)+1, input_length=40,
                     input_size=2, size_hidden=3, dmap=dmap,
                     extra_classifiers=extra_classifiers)

    # test save model
    A.save_model('saved_model')
    os.remove('saved_model')

    # test add pretrained model
    A.add_pretrained_model(A.model, dmap=dmap, classifiers=extra_classifiers)

    # test save model
    A.save_model('saved_model')
    os.remove('saved_model')

    # test generate training data from dictionary
    training_data = Training.generate_training_data(architecture, {'L1':5, 'L3l':10, 'L4r':10}, dmap=dmap, pad_to=40, classifiers=extra_classifiers)

    # test generate training data from math treebank?
    m = mathTreebank({'L1':5, 'L3l':10, 'L4r':10}, digits=np.arange(-10, 11))
    validation_data = Training.generate_training_data(architecture, m, dmap=dmap, pad_to=40, classifiers=extra_classifiers)
    # TODO implement this

    # test train
    A.train(training_data, batch_size=2, epochs=1, filename='temp', validation_data=validation_data)

    os.remove('temp1.h5')

    # test generate test data
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = Training.generate_test_data(A, languages=languages, dmap=dmap, digits=np.arange(-10, 11), pad_to=40, classifiers=extra_classifiers, test_separately=True)

    # test model testing
    for name, X, Y in test_data:
        A.model.evaluate(X, Y)

    # TODO test get model info?



