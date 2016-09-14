from __future__ import division

from collections import defaultdict, Counter

import data
import pickle
import core.myTheta as myTheta
import core.trainingRoutines as tr
import argparse
import sys




def install(thetaFile, d=0, noComparison = False):
  digits = [str(w) for w in range(-10, 11)]
  operators = ['+','-']
  ttb, htb, testtb = data.getTBs(digits, operators,noComparison=noComparison)

  try:
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
  except:
    dims = {'inside': d, 'outside': d, 'word': d, 'maxArity': 3, 'arity': 2}
    voc = ['UNKNOWN'] + digits + operators
    #voc = [str(w) for w in voc]
    grammar = {operator: {'(digit, ' + operator + ', digit)': 5} for operator in operators}
    theta = myTheta.Theta('RNN', dims, grammar, embeddings=None, vocabulary=voc)
    theta.extend4Classify(2,3,3*d)
  return theta, ttb, htb, testtb

def main(args):

  print args
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs','fixEmb','fixW']}
  noComp = args['noComp']
  if args['additive'] and not args['fixW']:
    print 'initializing with additive composition, but then train the function: nonsens, abort'
    sys.exit()

  theta, ttb, htb, testtb = install(args['pars'],d=args['word'],noComparison=noComp)

  datasets=[(ttb,htb,testtb)]
  phases=[10]
  hypers = [hyperParams]
  tr.alternate(theta, datasets,alt=phases, outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs'])
  #(theta, datasets, outDir, hyperParams, alt = [10, 1], n = 5):
  #tr.plainTrain(ttb, htb, testtb, hyperParams, theta, args['outDir'])


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
  parser.add_argument('-add','--additive', type=mybool, default=False, help='Whether the composition function is additive', required=False)
  parser.add_argument('-nc','--noComp', type=mybool, default=False, help='Whether the comparison layer is removed', required=False)

  args = vars(parser.parse_args())

  main(args)

