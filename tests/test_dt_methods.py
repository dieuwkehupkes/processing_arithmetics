import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from processing_arithmetics.sequential.architectures import ScalarPrediction, ComparisonTraining, DiagnosticTrainer, Seq2Seq, Training, DCgates
from keras.layers import GRU
import pickle
import os

@pytest.fixture(scope='module')
def data():
    data = {'digits': np.arange(-10, 11), 'operators': ['+', '-']}
    return data

@pytest.fixture()
def architecture(data):
    """Password fixture"""
    architecture = ScalarPrediction(digits=data['digits'], operators=data['operators'])
    # generate model
    np.random.seed(0)
    architecture.generate_model(recurrent_layer=GRU, input_length=20,
                  input_size=2, size_hidden=3)
    return architecture


@pytest.fixture()
def diagnostic_model(architecture, data):
    diagnostic_model = DiagnosticTrainer(digits=data['digits'], operators=data['operators'],
                                   classifiers=['subtracting', 'intermediate_locally', 'intermediate_recursively'],
                                   model=architecture.model)
    diagnostic_model.generate_model(recurrent_layer=GRU, input_length=20,
                                    input_size=2, size_hidden=3)
    return diagnostic_model


@pytest.fixture()
def diagnostic_model_no_classifiers(architecture, data):
    diagnostic_model = DiagnosticTrainer(digits=data['digits'], operators=data['operators'],
                                   classifiers=[],
                                   model=architecture.model)
    np.random.seed(0)
    diagnostic_model.generate_model(recurrent_layer=GRU, input_length=20,
                                    input_size=2, size_hidden=3)
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
    d = {'L1':5, 'L3l':5, 'L4r':5}
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

def test_equal(diagnostic_model_no_classifiers, architecture, data):
    # generate test data
    languages = {'L1':10, 'L2':15, 'L3':20}

    np.random.seed(0)
    test_data_sp = architecture.generate_test_data(data=languages, digits=data['digits'], test_separately=True)
    np.random.seed(0)
    test_data_dt = diagnostic_model_no_classifiers.generate_test_data(data=languages, digits=data['digits'], test_separately=True)

    # confirm models do equally well after initalisation
    test_dt = diagnostic_model_no_classifiers.test(test_data_dt, metrics=['mse'])
    test_sp = architecture.test(test_data_sp, metrics=['mse'])
    assert test_dt == test_sp

    # confirm models do equally well after training
    np.random.seed(5)
    training_data_sp = architecture.generate_training_data(data=languages, digits=data['digits'])
    np.random.seed(5)
    training_data_dt = diagnostic_model_no_classifiers.generate_training_data(data=languages, digits=data['digits'])

    # check training data is identical
    assert np.all(training_data_sp[0]['input'] == training_data_dt[0]['input'])

    # train models
    np.random.seed(0)
    architecture.train(training_data_sp, batch_size=2, epochs=1, filename='temp', optimizer='adam')
    os.remove('temp.h5')
    np.random.seed(0)
    diagnostic_model_no_classifiers.train(training_data_dt, batch_size=2, epochs=1, filename='temp', optimizer='adam')
    os.remove('temp.h5')

    # confirm models do equally well after training
    test_dt = diagnostic_model_no_classifiers.test(test_data_dt, metrics=['mse'])
    test_sp = architecture.test(test_data_sp, metrics=['mse'])
    assert test_dt == test_sp
    
