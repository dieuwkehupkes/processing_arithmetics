import pytest
import numpy as np
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.seqbased.architectures import ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq, Training
from keras.layers import SimpleRNN
import pickle
import os

def test_dmap():
    # generate architecture
    A = Training(np.arange(-10, 11), operators=['+','-'])

    # test if all elements are in dmap
    assert set(A.dmap.keys()) == set([str(i) for i in np.arange(-10,11)] + ['+','-','(',')'])

    # check if 0 is empty
    assert 0 not in A.dmap.values()

def test_ScalarPrediction_methods():
    _test_architecture_methods(ScalarPrediction)

def test_ComparisonTraining_methods():
    _test_architecture_methods(ComparisonTraining)

def test_DiagnosticClassifier_methods():
    _test_architecture_methods(DiagnosticClassifier, classifiers=['subtracting'])

def test_Seq2Seq_methods():
    _test_architecture_methods(Seq2Seq)

def test_recompile_new_metrics():
    # test if model can be tested with different metrics through recompilation
    digits = np.arange(-10, 11)
    operators = ['+', '-']

    # generate architecture, load dmap
    A = ScalarPrediction(digits=digits, operators=operators)

    # test build model
    A.generate_model(recurrent_layer=SimpleRNN, input_length=40,
                     input_size=2, size_hidden=3)

    # test generate test data
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = A.generate_test_data(data=languages, digits=np.arange(-10, 11), pad_to=40, test_separately=True)

    # test model testing
    A.test(test_data, metrics=['mean_squared_prediction_error'])


def _test_architecture_methods(architecture, **classifiers):
    """
    Test if methods of Training class still work.
    """

    digits = np.arange(-10, 11)
    operators = ['+', '-']

    # generate architecture, load dmap
    A = architecture(digits=digits, operators=operators)

    # test build model
    A.generate_model(recurrent_layer=SimpleRNN, input_length=40,
                     input_size=2, size_hidden=3,
                     **classifiers)

    # test save model
    A.save_model('saved_model')
    os.remove('saved_model')

    # test add pretrained model
    A.add_pretrained_model(A.model)

    # test save model
    A.save_model('saved_model')
    os.remove('saved_model')

    # test generate training data from dictionary
    training_data = A.generate_training_data({'L1':5, 'L3l':10, 'L4r':10}, pad_to=40)

    # test generate training data from math treebank
    m = MathTreebank({'L1':5, 'L3l':10, 'L4r':10}, digits=np.arange(-10, 11))
    validation_data = A.generate_training_data(m, pad_to=40)

    # test train
    if os.path.exists('temp1.h5'):
        os.remove('temp1.h5')
    A.train(training_data, batch_size=2, epochs=1, filename='temp', validation_data=validation_data)

    os.remove('temp1.h5')

    # test generate test data
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = A.generate_test_data(data=languages, digits=np.arange(-10, 11), pad_to=40, test_separately=True)

    # test model testing
    A.test(test_data)

    # TODO test get model info?



