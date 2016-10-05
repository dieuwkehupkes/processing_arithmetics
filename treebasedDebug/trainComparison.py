from __future__ import division

from collections import defaultdict, Counter

import data
import pickle
import core.myTheta as myTheta
import core.trainingRoutines as tr
import argparse
import sys
import core.gradient_check as check

def installTheta(thetaFile, seed, d, noComparison):
  if thetaFile != '':
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
    print 'initialized theta from file:', thetaFile
    if noComparison and ('comparison','M') in theta.keys():
      theta.extend4Classify(2, 3, -1)
    if not noComparison and ('comparison', 'M') not in theta.keys():
      theta.extend4Classify(2, 3, 3 * d)

    if '+' not in theta[('word',)].voc:
      theta[('word',)].extendVocabulary(['+','-'])
      theta[('word',)]['+']=theta[('word',)]['plus']
      theta[('word',)]['-'] = theta[('word',)]['minus']

  else:
    print 'initializing theta from scratch'
    dims = {'inside': d, 'outside': d, 'word': d, 'maxArity': 3, 'arity': 2}
    voc = ['UNKNOWN'] + [str(w) for w in data.arithmetics.ds] + data.arithmetics.ops
    grammar = {operator: {'(digit, ' + operator + ', digit)': 5} for operator in data.arithmetics.ops}
    theta = myTheta.Theta('RNN', dims, grammar, embeddings=None, vocabulary=voc, seed = seed)
    if noComparison: theta.extend4Classify(2,3,-1)
    else: theta.extend4Classify(2,3,3*d)
  #theta.extend4Prediction(-1)
  return theta


def main(args):
  print args
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs']}
  hyperParamsCompare = hyperParams.copy()
  hyperParamsPredict = hyperParams.copy()
  hyperParamsCompare['fixEmb'] = False
  hyperParamsCompare['fixW'] = False
  hyperParamsPredict['fixEmb'] = False#True
  hyperParamsPredict['fixW'] = False#True

  theta = installTheta(args['pars'],seed=args['seed'],d=args['word'],noComparison=args['noComp'])
  datasets = [data.getComparisonTBs(args['seed'], args['noComp'])]

  #nw,tar = predictData['train'].examples[0]
  #check.gradientCheck(theta,nw,tar)

#  datasets=[predictData,compareData]
#  hypers = [hyperParamsPredict, hyperParamsCompare]
#  phases=[10,5] #number of epochs for either phase
#  names = ['scalarprediction','compare expressions']

  # datasets = [predictData]
  # hypers = [hyperParamsPredict]
  # phases = [25]  # number of epochs for either phase
  # names = ['scalarprediction']

  #datasets = [compareData]
  hypers = [hyperParamsCompare]
  phases = [25]  # number of epochs for either phase
  names = ['compare expressions']

  tr.alternate(theta, datasets,alt=phases, outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs'], names=names)
  #predictData['train'].evaluate(theta,'train',n=20, verbose=True)

def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  elif string in ['T', 't', 'true', 'True']: return True
  else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-o','--outDir', type=str, help='Output dir to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, default='', help='File with pickled theta', required=False)
  # network hyperparameters:
  parser.add_argument('-dwrd','--word', type=int, default = 2, help='Dimensionality of leaves (word nodes)', required=False)
  # training hyperparameters:
  parser.add_argument('-s', '--seed', type=int, help='Random seed to be used', required=True)
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  parser.add_argument('-ada','--ada', type=mybool, help='Whether adagrad is used', required=True)
  parser.add_argument('-c','--cores', type=int, default=1,help='The number of parallel processes', required=False)
  parser.add_argument('-nc','--noComp', type=mybool, default=True, help='Whether the comparison layer is removed', required=False)


  args = vars(parser.parse_args())

  main(args)

