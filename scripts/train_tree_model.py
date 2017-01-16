from __future__ import division
from matplotlib import pyplot as plt

from processing_arithmetics.tree import data
import pickle
from processing_arithmetics.tree import myTheta as myTheta
from processing_arithmetics.tree import trainingRoutines as tr
from processing_arithmetics.tree import Optimizer
from processing_arithmetics.arithmetics import treebanks
import argparse
from collections import defaultdict
import os

''' instantiate parameters (theta object): obtain theta from file or create a new theta'''
def installTheta(thetaFile, seed, d, comparison):
    if thetaFile != '':
        with open(thetaFile, 'rb') as f:
            theta = pickle.load(f)
        print 'Initialized model from file:', thetaFile
        if ('classify','B') not in theta: theta.extend4Classify(2, 3, comparison)

        # legacy; in theta from older versions of the code 'plus' and 'minus' in the vocabulary
        if '+' not in theta[('word',)].voc:
            theta[('word',)].extendVocabulary(['+','-'])
            theta[('word',)]['+'] = theta[('word',)]['plus']
            theta[('word',)]['-'] = theta[('word',)]['minus']

    else:

        dims = {'inside': d[0], 'word': d[1], 'minArity': 3, 'maxArity': 3}
        voc = ['UNKNOWN'] + [str(w) for w in data.arithmetics.ds] + treebanks.ops
        theta = myTheta.Theta(dims=dims, embeddings=None, vocabulary=voc, seed = seed)
        theta.extend4Classify(2,3,comparison)
        print 'Initialized model from scratch, dims:',dims
    theta.extend4Prediction(-1)
    return theta


def main(args):
    print args
    hyperParams = {k:args[k] for k in ['bSize']}
    hyperParams['toFix'] = [] # a selection of parameters can be fixed, e.g. the word embeddings

    # initialize theta (object with model parameters)
    theta = installTheta(args['pars'],seed=args['seed'],d=(args['dim'],args['dword']),comparison=args['comparison'])

    # initialize optimizer with learning rate (other hyperparams: default values)
    opt = args['optimizer']
    if opt == 'adagrad': optimizer = Optimizer.Adagrad(theta,lr = args['learningRate'])
    elif opt == 'adam': optimizer = Optimizer.Adam(theta,lr = args['learningRate'])
    elif opt == 'sgd': optimizer = Optimizer.SGD(theta,lr = args['learningRate'])
    else: raise RuntimeError("No valid optimizer chosen")

    # generate training and heldout data
    dataset = data.getTBs(seed=args['seed'], kind='comparison', comparisonLayer=args['comparison'])

    # start training
    # returns the scores for convergence analysis
    # prints intermediate results

    nEpochs = args['nEpochs']
    f = args['storageFreq']
    evals = defaultdict(list)


    for i in range(nEpochs//f):
      # store model parameters,
      # train f epochs and run an evaluation
        outFile = os.path.join(args['outDir'], 'startEpoch' + str(i*f) + '.theta.pik')
        tr.storeTheta(optimizer.theta, outFile)
        for name, tb in dataset.iteritems():
            print('Evaluation on ' + name + ' data')
            tb.evaluate(optimizer.theta, verbose=1)
        thisEvals = tr.plainTrain(optimizer, dataset['train'], dataset['heldout'], hyperParams, nEpochs=f, nStart=i * f)
        for name in thisEvals.iterkeys(): evals[name]+= thisEvals[name]

    if nEpochs%f!=0:
        evals = tr.plainTrain(optimizer, dataset['train'], dataset['heldout'], hyperParams, nEpochs=nEpochs%f, nStart=(i+1)* f)
        for name in thisEvals.iterkeys(): evals[name] += thisEvals[name]

    # store final model parameters and run final evaluation
    for name, tb in dataset.iteritems():
        print('Evaluation on '+name+' data ('+str(len(tb.examples))+' examples)')
        tb.evaluate(optimizer.theta, verbose=1)


    # create convergence plot
    for name, eval in evals.items():
        toplot = [e[key] for e in eval for key in e if 'loss' in key]
        plt.plot(xrange(len(toplot)), toplot,label=name)
    plt.legend()
    plt.title([key for key in eval[0].keys() if 'loss' in key][0])
    plt.savefig(os.path.join(args['outDir'],'convergencePlot.png'))

def mybool(string):
    if string in ['F', 'f', 'false', 'False']: return False
    elif string in ['T', 't', 'true', 'True']: return True
    else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    # storage:
    parser.add_argument('-o','--outDir', type=str, help='Output dir to store model', required=True)
    parser.add_argument('-p','--pars', type=str, default='', help='Existing model file', required=False)
    # network hyperparameters:
    parser.add_argument('-c','--comparison', type=int, default=0, help='Dimensionality of comparison layer (0 is no layer)', required=False)
    parser.add_argument('-d','--dim', type=int, default = 2, help='Dimensionality of internal representations', required=False)
    parser.add_argument('-dw','--dword', type=int, default = 2, help='Dimensionality of word embeddings', required=False)
    # training hyperparameters:
    parser.add_argument('-opt', '--optimizer', type=str, default='sgd', choices=['sgd', 'adagrad', 'adam'], help='Optimization scheme', required=False)
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed to be used', required=False)
    parser.add_argument('-n','--nEpochs', type=int, default=100, help='Maximal number of epochs to train', required=False)
    parser.add_argument('-f', '--storageFreq', type=int, default=10, help='Model is stored after every f epochs',
                        required=False)
    parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
    parser.add_argument('-l','--lambda', type=float, default=0.0001, help='Regularization parameter lambdaL2', required=False)
    parser.add_argument('-lr','--learningRate', type=float, default=0.01, help='Learning rate parameter', required=False)


    args = vars(parser.parse_args())

    main(args)

