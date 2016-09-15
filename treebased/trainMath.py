from __future__ import division

from collections import defaultdict, Counter

import data
import pickle
import core.myTheta as myTheta
import core.trainingRoutines as tr
import argparse
import sys
import core.gradient_check as check



def install(thetaFile, d=0, noComparison = False):
  digits = [str(w) for w in range(-10, 11)]
  operators = ['+','-']


  compareData,predictData = data.getTBs(digits, operators,noComparison=noComparison)

  try:
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
  except:
    dims = {'inside': d, 'outside': d, 'word': d, 'maxArity': 3, 'arity': 2}
    voc = ['UNKNOWN'] + digits + operators
    grammar = {operator: {'(digit, ' + operator + ', digit)': 5} for operator in operators}
    theta = myTheta.Theta('RNN', dims, grammar, embeddings=None, vocabulary=voc)
    if noComparison: theta.extend4Classify(2,3,-1)
    else: theta.extend4Classify(2,3,3*d)
    theta.extend4Prediction()
  return theta, compareData, predictData

def main(args):

  print args
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs']}
  hyperParamsCompare = hyperParams.copy()
  hyperParamsPredict = hyperParams.copy()
  hyperParamsCompare['fixEmb'] = False
  hyperParamsCompare['fixW'] = False
  hyperParamsPredict['fixEmb'] = False#True
  hyperParamsPredict['fixW'] = False#True

  theta, compareData,predictData = install(args['pars'],d=args['word'],noComparison=args['noComp'])

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

  datasets = [compareData]
  hypers = [hyperParamsPredict]
  phases = [25]  # number of epochs for either phase
  names = ['compare expressions']

  tr.alternate(theta, datasets,alt=phases, outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs'], names=names)
  predictData['train'].evaluate(theta,'train',n=20, verbose=True)

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
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-a','--alpha', type=float, help='Learning rate parameter alpha', required=True)
  parser.add_argument('-ada','--ada', type=mybool, help='Whether adagrad is used', required=True)
  parser.add_argument('-c','--cores', type=int, default=1,help='The number of parallel processes', required=False)
  parser.add_argument('-fw','--fixEmb', type=mybool, default=False, help='Whether the word embeddings are fixed', required=False)
  parser.add_argument('-fc','--fixW', type=mybool, default=False, help='Whether the composition function is fixed', required=False)
  parser.add_argument('-nc','--noComp', type=mybool, default=True, help='Whether the comparison layer is removed', required=False)


  args = vars(parser.parse_args())

  main(args)

