from __future__ import division

from collections import defaultdict, Counter

import data
import pickle
import core.myTheta as myTheta
import core.trainingRoutines as tr
import argparse
import sys
import copy
import core.gradient_check as check
import core.Optimizer as opt


def install(thetaFile, d, noComparison, predictH):
  digits = [str(w) for w in range(-10, 11)]
  operators = ['+','-']


  compareData,predictData = data.getTBs(digits, operators,noComparison=noComparison)

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
    voc = ['UNKNOWN'] + digits + operators
    grammar = {operator: {'(digit, ' + operator + ', digit)': 5} for operator in operators}
    theta = myTheta.Theta('RNN', dims, grammar, embeddings=None, vocabulary=voc)
    if noComparison: theta.extend4Classify(2,3,-1)
    else: theta.extend4Classify(2,3,3*d)
  if predictH: theta.extend4Prediction()
  else: theta.extend4Prediction(-1)
  return theta, compareData, predictData

def main(args):

  print args
  hyperParams={k:args[k] for k in ['bSize','lambda','alpha','ada','nEpochs']}
  hyperParams['tofix']=[]


  classifynames = ['classify','comparison'] #'('classify', 'M'), ('classify', 'B'),('comparison', 'B'), ('comparison', 'M')]
  predictnames = ['predict','predictH'] #('predict', 'B'),('predict', 'M'), ('predictH', 'M'), ('predictH', 'B')]
  compositionnames =  ['composition']#('composition', '  # X#', '(#X#)', 'I', 'M'), ('composition', '#X#', '(#X#)', 'I', 'B'),('composition','#X#', '(#X#, #X#, #X#)', 'I', 'B'),('composition', '#X#', '(#X#, #X#)', 'I', 'B'), ('composition', '#X#', '(#X#, #X#)', 'I', 'M'), ('composition', '#X#', '(#X#, #X#,  # X#)', 'I', 'M')]
  embnames = ['word']#'('word',)]



  theta, compareData,predictData = install(args['pars'],d=args['word'],noComparison=args['noComp'],predictH=args['predictH'])
  if args['ada']: optimizer = opt.Adagrad(theta,)
  else: optimizer = opt.SGD(theta,)

  datasets = [compareData, predictData]
  names = ['compare expressions', 'scalarprediction']
  hyperParamsCompare = copy.deepcopy(hyperParams)
  hyperParamsPredict = copy.deepcopy(hyperParams)

  for k, dataset in zip (names,datasets):
    print 'Evaluating initial theta on', k
    for name, d in dataset.iteritems():
      d.evaluate(theta, name,verbose=args['verbose'])


  verbose = args['verbose']
  #nw,tar = predictData['train'].examples[0]
  #check.gradientCheck(theta,nw,tar)

  if args['kind']=='a1': #train alternating, but only train embeddings/ composition function during comparison training
    hyperParamsCompare['tofix'] += compositionnames + embnames
    datasets=[predictData,compareData]
    hypers = [hyperParamsPredict, hyperParamsCompare]
    phases=[10,5] #number of epochs for either phase
    names = ['scalarprediction','compare expressions']
  elif args['kind'] == 'a2': #train alternating, but only train embeddings/ composition function during prediction training
    hyperParamsPredict['tofix'] += compositionnames + embnames
    datasets = [compareData,predictData]
    hypers = [hyperParamsCompare,hyperParamsPredict]
    phases = [10, 5]  # number of epochs for either phase
    names = ['compare expressions','scalarprediction']
  elif args['kind'] == 'a3': #train alternating, and train embeddings/ composition function during both training phases
      datasets = [predictData, compareData]
      hypers = [hyperParamsPredict, hyperParamsCompare]
      phases = [3, 3]  # number of epochs for either phase
      names = ['scalarprediction', 'compare expressions']
  elif args['kind']=='c':
    datasets=[compareData]
    hypers = [hyperParamsCompare]
    phases=[10] #number of epochs per round
    names = ['compare expressions']
  elif args['kind'] == 's1': #train scalar prediction but leave embeddings/ composition as is
    hyperParamsPredict['tofix'] += compositionnames + embnames
    datasets = [predictData]
    hypers = [hyperParamsPredict]
    phases = [10]  # number of epochs per round
    names = ['scalarprediction']
  elif args['kind'] == 's2':
    datasets = [predictData]
    hypers = [hyperParamsPredict]
    phases = [10]  # number of epochs per round
    names = ['scalarprediction']
  else:
    print 'no kind!',args['kind']
#    sys.exit()

  tr.alternate(optimizer, args['outDir'], datasets, hypers, phases, n = args['nEpochs'], names = names, verbose = verbose)
  #predictData['train'].evaluate(theta,'train',n=20, verbose=True)

def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  elif string in ['T', 't', 'true', 'True']: return True
  else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-k', '--kind', type=str, help='scalar prediction (sp) or comparison (c) or alternating (a1/a2/a3)', required=True)
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
  parser.add_argument('-nc','--noComp', type=mybool, default=True, help='Whether the comparison layer is removed', required=False)
  parser.add_argument('-ph', '--predictH', type=mybool, default=True, help='Whether there is a hidden layer for scalar prediction',
                      required=False)
  parser.add_argument('-v', '--verbose', type=mybool, default=False, help='Whether a lot of output is printed',
                      required=False)

  args = vars(parser.parse_args())

  main(args)

