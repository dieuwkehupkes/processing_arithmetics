from __future__ import division
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

from processing_arithmetics.tree import data
import pickle
from processing_arithmetics.tree.core import myTheta as myTheta
from processing_arithmetics.tree.core import trainingRoutines as tr
from processing_arithmetics.tree.core import Optimizer
from processing_arithmetics.arithmetics import treebanks
import argparse
import sys
import os

''' instantiate parameters (theta object): obtain theta from file or create a new theta'''
def installTheta(thetaFile, seed, d, comparison):
    if thetaFile != '':
        with open(thetaFile, 'rb') as f:
            theta = pickle.load(f)
        print 'initialized theta from file:', thetaFile
        if ('classify','B') not in theta: theta.extend4Classify(2, 3, comparison)

        # legacy; in theta from older versions of the code 'plus' and 'minus' in the vocabulary
        if '+' not in theta[('word',)].voc:
            theta[('word',)].extendVocabulary(['+','-'])
            theta[('word',)]['+'] = theta[('word',)]['plus']
            theta[('word',)]['-'] = theta[('word',)]['minus']

    else:
        print 'initializing theta from scratch'
        dims = {'inside': d, 'outside': d, 'word': d, 'minArity': 3, 'maxArity': 3}
        voc = ['UNKNOWN'] + [str(w) for w in data.arithmetics.ds] + treebanks.ops
        theta = myTheta.Theta(dims=dims, embeddings=None, vocabulary=voc, seed = seed)
        theta.extend4Classify(2,3,comparison)
    theta.extend4Prediction(-1)
    return theta


def main(args):
    print args
    hyperParams = {k:args[k] for k in ['bSize']}
    hyperParams['toFix'] = []
    hyperParamsCompare = hyperParams.copy()
    hyperParamsPredict = hyperParams.copy()

    # initialize theta (object with model parameters)
    theta = installTheta(args['pars'],seed=args['seed'],d=args['word'],comparison=args['comparison'])

    # initialize optimizer with learning rate (other hyperparams: default values)
    opt = args['optimizer']
    if opt == 'adagrad': optimizer = Optimizer.Adagrad(theta,lr = args['learningRate'])
    elif opt == 'adam': optimizer = Optimizer.Adam(theta,lr = args['learningRate'])
    elif opt == 'sgd': optimizer = Optimizer.SGD(theta,lr = args['learningRate'])
    else: raise RuntimeError("No valid optimizer chosen")

    '''
    Training in phases: train for a phase on a certain task, then perform a complete evaluation
    Phases can be used to alternate tasks to train on, possibly fixing some of the model parameters
    NB: this feature is not really used in the current experiments
    '''
    q = 10
    if q > args['nEpochs']:
        q = args['nEpochs']//2

    kind = args['kind']
    if kind == 'c':
        hypers = [hyperParamsCompare]
        names = ['comparison']
        phases = [q]
    elif kind == 'p':
        hypers = [hyperParamsPredict]
        names = ['prediction']
        phases = [q]
    elif kind == 'a':
        hypers = [hyperParamsPredict, hyperParamsCompare]
        names = ['prediction', 'comparison']
        phases = [q//2,q//2]
    else: sys.exit()

    # generate training and evaluation data for the tasks to be trained on
    datasets = [data.getTBs(seed=args['seed'], kind=name, comparison=args['comparison']) for name in names]

    # start training
    # returns the scores for convergence analysis
    # prints intermediate results
    evals = tr.alternate(optimizer, datasets, alt=phases,outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs']//q, names=names)

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
    parser.add_argument('-k', '--kind', type=str, choices=['c','p','a'], help='Kind of networks: [c]omparison, [p]rediction or [a]lternating', required=True)
    parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
    # data:
    parser.add_argument('-o','--outDir', type=str, help='Output dir to store pickled theta', required=True)
    parser.add_argument('-p','--pars', type=str, default='', help='File with pickled theta', required=False)
    # network hyperparameters:
    parser.add_argument('-c','--comparison', type=int, default=0, help='Dimensionality of comparison layer (0 is no layer)', required=False)
    parser.add_argument('-dwrd','--word', type=int, default = 2, help='Dimensionality of leaves (word nodes)', required=False)
    # training hyperparameters:
    parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adagrad', 'adam'], help='Optimization scheme', required=True)
    parser.add_argument('-s', '--seed', type=int, help='Random seed to be used', required=True)
    parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train', required=True)
    parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
    parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
    parser.add_argument('-lr','--learningRate', type=float, help='Learning rate parameter', required=True)


    args = vars(parser.parse_args())

    main(args)

