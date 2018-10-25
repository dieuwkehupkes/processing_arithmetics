import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from processing_arithmetics.sequential.architectures import ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq, Training, DCgates
from keras.layers import GRU
import pickle
import os

@pytest.fixture(scope='module')
def data():
    data = {'digits': np.arange(-10, 11), 'operators': ['+', '-']}
    return data

@pytest.fixture(params=[
    ScalarPrediction,
    # ComparisonTraining,
    Seq2Seq
])
def architecture(request, data):
    """Password fixture"""
    architecture = request.param(digits=data['digits'], operators=data['operators'])
    # generate model
    architecture.generate_model(recurrent_layer=GRU, input_length=20,
                  input_size=2, size_hidden=3)
    return architecture


@pytest.fixture(params=[
    DiagnosticClassifier
])
def diagnostic_model(request, architecture, data):
    diagnostic_model = request.param(digits=data['digits'], operators=data['operators'],
                                   classifiers=['subtracting', 'intermediate_locally', 'intermediate_recursively'],
                                   model=architecture.model)
    return diagnostic_model


def test_save_model(diagnostic_model, data):
    diagnostic_model.save_model('saved_model')
    os.remove('saved_model')


def test_add_pretrained_model(diagnostic_model, data):
    diagnostic_model.save_model('saved_model')
    diagnostic_model.add_pretrained_model('saved_model')
    os.remove('saved_model')


def test_training(diagnostic_model, data):
    # test generate training data from dict
    d = {'L1':5, 'L3l':10, 'L4r':10}
    training_data = diagnostic_model.generate_training_data(d)

    # test generate training data from treebank
    m = MathTreebank(d, digits=data['digits'])
    val_data = diagnostic_model.generate_training_data(m)

    if os.path.exists('temp.h5'):
        os.remove('temp.h5')
    diagnostic_model.train(training_data, batch_size=2, epochs=1, filename='temp', 
                   validation_data=val_data, optimizer='adam')

    os.remove('temp.h5')


def test_testing(diagnostic_model, data):
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = diagnostic_model.generate_test_data(data=languages, digits=data['digits'], test_separately=True)

    # test model testing
    diagnostic_model.test(test_data, metrics=['mse', 'mspe', 'binary_accuracy'])
