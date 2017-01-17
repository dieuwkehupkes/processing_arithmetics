from __future__ import division

from processing_arithmetics.tree import data, myTheta
import pickle
from processing_arithmetics.tree import trainingRoutines as ctr
from processing_arithmetics.tree import predictionTraining as ptr
from processing_arithmetics.arithmetics import treebanks
import argparse
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
    #print args
    # initialize theta (object with model parameters)
    theta = installTheta(args['parsC'],seed=args['seed'],d=(args['dim'],args['dword']),comparison=args['comparison'])

    # generate training and heldout data for comparion training and train model
    datasetC = data.data4comparison(seed=args['seed'], comparisonLayer=args['comparison'],debug=args['debug'])
    comparisonArgs={k[:-1]:v for (k,v) in args.iteritems() if k[-1]=='C'}
    comparisonArgs['outDir']=args['outDir']
    print('Comparison training:'+ str(comparisonArgs))
    ctr.trainComparison(comparisonArgs, theta, datasetC)

    # generate training and heldout data for prediction training and train model
    datasetP = data.data4prediction(theta, seed=args['seed'],debug=args['debug'])
    predictionArgs = {k[:-1]: v for (k, v) in args.iteritems() if k[-1] == 'P'}
    predictionArgs['outDir'] = args['outDir']
    print('Prediction training:' + str(predictionArgs))
    ptr.trainPrediction(predictionArgs,datasetP,str(001))


def mybool(string):
    if string in ['F', 'f', 'false', 'False']: return False
    elif string in ['T', 't', 'true', 'True']: return True
    else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('-debug','--debug',type=mybool, default=False, required=False)
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed to be used', required=False)
    # storage:
    parser.add_argument('-o','--outDir', type=str, help='Output dir to store models', required=True)
    parser.add_argument('-pc','--parsC', type=str, default='', help='Existing model file (TreeRNN)', required=False)
    parser.add_argument('-pp', '--parsP', type=str, default='', help='Existing model file (Keras)', required=False)
    # network hyperparameters TreeRNN:
    parser.add_argument('-c','--comparison', type=int, default=0, help='Dimensionality of comparison layer (0 is no layer)', required=False)
    parser.add_argument('-d','--dim', type=int, default = 2, help='Dimensionality of internal representations', required=False)
    parser.add_argument('-dw','--dword', type=int, default = 2, help='Dimensionality of word embeddings', required=False)
    # network hyperparameters Prediction:
    parser.add_argument('-dh', '--dHiddenP', type=int, default=0, help='Dimensionality of hidden layer (0 is no layer)', required=False)
    parser.add_argument('-loss', '--lossP',choices=['mse', 'mae'], default='mse', help='Loss function for prediction', required=False)
    # training hyperparameters comparison:
    parser.add_argument('-optC', '--optimizerC', type=str, default='sgd', choices=['sgd', 'adagrad', 'adam'], help='Optimization scheme for comparison training', required=False)
    parser.add_argument('-nc','--nEpochsC', type=int, default=100, help='Number of epochs for comparison training', required=False)
    parser.add_argument('-bc','--bSizeC', type=int, default = 50, help='Batch size for comparison training', required=False)
    parser.add_argument('-f', '--storageFreqC', type=int, default=10, help='Model is evaluated and stored after every f epochs', required=False)
    parser.add_argument('-lc','--lambdaC', type=float, default=0.0001, help='Regularization parameter lambdaL2', required=False)
    parser.add_argument('-lrc','--learningRateC', type=float, default=0.01, help='Learning rate parameter', required=False)

    # training hyperparameters prediction:
    parser.add_argument('-np', '--nEpochsP', type=int, default=100, help='Number of epochs for prediction training',
                        required=False)
    parser.add_argument('-bp', '--bSizeP', type=int, default=50, help='Batch size for prediction training', required=False)
    parser.add_argument('-lp','--lambdaP', type=float, default=0.0001, help='Regularization parameter lambdaL2', required=False)
    parser.add_argument('-lrp','--learningRateP', type=float, default=0.01, help='Learning rate parameter', required=False)


    args = vars(parser.parse_args())

    main(args)

