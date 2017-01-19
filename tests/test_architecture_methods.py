import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from processing_arithmetics.sequential.architectures import ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq, Training
from keras.layers import SimpleRNN
import pickle
import os

@pytest.fixture(scope='module')
def data():
    data = {'digits': np.arange(-10, 11), 'operators': ['+', '-']}
    return data

@pytest.fixture(params=[
    ScalarPrediction,
    ComparisonTraining,
    Seq2Seq
])
def architecture(request, data):
    """Password fixture"""
    architecture = request.param(digits=data['digits'], operators=data['operators'])
    # generate model
    architecture.generate_model(recurrent_layer=SimpleRNN, input_length=40,
                  input_size=2, size_hidden=3)
    return architecture


@pytest.fixture()
def dc_model(architecture, data):
    dc_model = DiagnosticClassifier(digits=data['digits'], operators=data['operators'],
                                   classifiers=['subtracting', 'intermediate_locally', 'intermediate_recursively'],
                                   model=architecture.model)
    return dc_model


# test architecture methods
def test_save_model(architecture, data):
    architecture.save_model('saved_model')
    os.remove('saved_model')


def test_add_pretrained_model(architecture, data):
    architecture.save_model('saved_model')
    architecture.add_pretrained_model('saved_model')
    os.remove('saved_model')


def test_training(architecture, data):
    # test generate training data from dict
    d = {'L1':5, 'L3l':10, 'L4r':10}
    training_data = architecture.generate_training_data(d)

    # test generate training data from treebank
    m = MathTreebank(d, digits=data['digits'])
    val_data = architecture.generate_training_data(m)

    if os.path.exists('temp.h5'):
        os.remove('temp.h5')
    architecture.train(training_data, batch_size=2, epochs=1, filename='temp', 
                   validation_data=val_data, optimizer='adam')

    os.remove('temp.h5')


def test_testing(architecture, data):
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = architecture.generate_test_data(data=languages, digits=data['digits'], test_separately=True)

    # test model testing
    architecture.test(test_data, metrics=['mse', 'mspe', 'binary_accuracy'])


# test dc methods
def test_save_dc_model(dc_model, data):
    dc_model.save_model('saved_model')
    os.remove('saved_model')


def test_add_pretrained_model_dc(dc_model, data):
    dc_model.save_model('saved_model')
    dc_model.add_pretrained_model('saved_model')
    os.remove('saved_model')


def test_dc_training(dc_model, data):
    # test generate training data from dict
    d = {'L1':5, 'L3l':10, 'L4r':10}
    training_data = dc_model.generate_training_data(d)

    # test generate training data from treebank
    m = MathTreebank(d, digits=data['digits'])
    val_data = dc_model.generate_training_data(m)

    if os.path.exists('temp.h5'):
        os.remove('temp.h5')
    dc_model.train(training_data, batch_size=2, epochs=1, filename='temp', 
                   validation_data=val_data, optimizer='adam')

    os.remove('temp.h5')


def test_dc_testing(dc_model, data):
    languages = {'L1':10, 'L2':15, 'L3':20}
    test_data = dc_model.generate_test_data(data=languages, digits=data['digits'], test_separately=True)

    # test model testing
    dc_model.test(test_data, metrics=['mse', 'mspe', 'binary_accuracy'])



# test dmap
def test_dmap(data):
    # generate architecture
    A = Training(digits=data['digits'], operators=data['operators'])

    # test if all elements are in dmap
    assert set(A.dmap.keys()) == set([str(i) for i in np.arange(-10,11)] + ['+','-','(',')'])

    # check if 0 is empty
    assert 0 not in A.dmap.values()

 
if __name__ == '__main__':
    pytest.main([__file__])
