import pytest
import numpy as np

from processing_arithmetics.treebased import data, myTheta
from processing_arithmetics.sequential.architectures import ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq, Training
from keras.layers import SimpleRNN
import pickle
import os

def test_model_storage():
    model = myTheta.installTheta('', 0, [2,2], 5)
    #print model[('comparison','B')]
    assert model[('comparison','B')].shape[0] == 5
    assert model[('comparison', 'M')].shape == (5,4)
    modelfile = 'modeltmp.pik'
    with open(modelfile, 'wb') as f: pickle.dump(model, f)

    model = myTheta.installTheta('', 0, [0,0], 0)
    assert ('comparison',) not in model
    model = myTheta.installTheta(modelfile, 0, 2, 5)
    assert model[('comparison', 'M')].shape == (5, 4)

def _numgr(nw, target, theta, item, itemgrad, epsilon = 0.0001):
    it = np.nditer(item, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
    #for i in range(len(item)):
        old = item[i]
        item[i]= old + epsilon
        errorPlus = nw.error(theta, target, True)
        item[i] = old - epsilon
        errorMin = nw.error(theta, target, True)
        item[i] = old
        if errorPlus==errorMin: itemgrad[i]=0
        else: itemgrad[i] = (errorPlus - errorMin) / (2 * epsilon)
        it.iternext()

def test_gradient():
    theta = myTheta.installTheta('', 0, [2,2], 5)
    dataset = data.data4comparison(0,True,True)
    for nw, target in dataset['train'].examples:
        grad = theta.gradient()
        _ = nw.train(theta, grad, activate=True, target=target)
        numgrad = theta.gradient()
        for name in theta.keys():
            if name == ('word',):  # True
                for word in theta[name].keys():
                    _numgr(nw, target, theta, theta[name][word], numgrad[name][word])
            else:
                _numgr(nw, target, theta, theta[name], numgrad[name])
        gradflat = np.array([])
        numgradflat = np.array([])
        for name in theta.keys():
            if name == ('word',):  # True
                ngr = np.concatenate([numgrad[name][word] for word in theta[name].keys()])
                gr = np.concatenate([grad[name][word] for word in theta[name].keys()])
            else:
                ngr = np.reshape(numgrad[name], -1)
                gr = np.reshape(grad[name], -1)
            if np.array_equal(gr, ngr):
                diff = 0
            else:
                diff = np.linalg.norm(ngr - gr) / (np.linalg.norm(ngr) + np.linalg.norm(gr))
            assert diff < 0.00001
            gradflat = np.append(gradflat, gr)
            numgradflat = np.append(numgradflat, ngr)
        assert np.linalg.norm(numgradflat - gradflat) / (np.linalg.norm(numgradflat) + np.linalg.norm(gradflat)) < 0.00001

if __name__ == '__main__':
    pytest.main([__file__])
