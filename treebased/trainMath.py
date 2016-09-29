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


def installTheta(thetaFile, seed, d, noComparison, predictH):
  if thetaFile != '':
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
    print 'initialized theta from file:', thetaFile
    if noComparison and ('comparison','M') in theta.keys():
      theta.extend4Classify(2, 3, -1)
    if not noComparison and ('comparison', 'M') not in theta.keys():
      theta.extend4Classify(2, 3, 3 * d)

    # if '+' not in theta[('word',)].voc:
    #   theta[('word',)].extendVocabulary(['+','-'])
    #   theta[('word',)]['+']=theta[('word',)]['plus']
    #   theta[('word',)]['-'] = theta[('word',)]['minus']

  else:
    print 'initializing theta from scratch'
    dims = {'inside': d, 'outside': d, 'word': d, 'maxArity': 3, 'arity': 2}
    voc = ['UNKNOWN'] + data.digits + data.operators
    grammar = {operator: {'(digit, ' + operator + ', digit)': 5} for operator in data.operators}
    theta = myTheta.Theta('RNN', dims, grammar, embeddings=None, vocabulary=voc, seed = seed)
    if noComparison: theta.extend4Classify(2,3,-1)
    else: theta.extend4Classify(2,3,3*d)
  if predictH: theta.extend4Prediction()
  else: theta.extend4Prediction(-1)
  return theta

def evaluateThetas(predictData):
  filenames = ['phase0startEpoch0.theta.pik','phase0startEpoch10.theta.pik','phase0startEpoch20.theta.pik','phase0startEpoch25.theta.pik','phase0startEpoch30.theta.pik','phase0startEpoch40.theta.pik','phase0startEpoch50.theta.pik','phase0startEpoch60.theta.pik','phase0startEpoch70.theta.pik','phase0startEpoch80.theta.pik','phase0startEpoch90.theta.pik','phase0startEpoch100.theta.pik','phase0startEpoch110.theta.pik','phase0startEpoch120.theta.pik','phase0startEpoch130.theta.pik','phase0startEpoch140.theta.pik','phase0startEpoch150.theta.pik','phase0startEpoch160.theta.pik','phase0startEpoch170.theta.pik','phase0startEpoch180.theta.pik','phase0startEpoch190.theta.pik']
  for filename in filenames:
    print filename
    with open('trainedModels/LILSP/001/'+filename, 'rb') as f:
      theta = pickle.load(f)
    for name, d in predictData.iteritems():
      try: d.evaluate(theta, name)
      except:
        for lan, land in dataset.iteritems():
           land.evaluate(theta, name+'_'+lan)

    predictData['heldout'].evaluate(theta, '', verbose=True, n=10)
  return theta



def fullEvaluation(theta, seed, noComparison, predictH):
  evals = {'comparison':{}, 'prediction':{}}
  for lan, compareTB, predictTB in data.getTestTBs(seed, noComparison, predictH):
    evals['comparison'][lan] = compareTB.evaluate(theta,verbose=0)
    evals['prediction'][lan] = predictTB.evaluate(theta,verbose=0)
  for name, eval in evals.iteritems():
    print 'Evaluation on', name, 'per language'
    print  'lan\t\tloss\t\taccuracy\t\t( evaluation metric)'
    for val in sorted([[key]+[str(v) for v in value] for key, value in eval.iteritems()]): print '\t\t'.join(val)


def main(args):

  seed = 0

  print args
  hyperParams={k:args[k] for k in ['bSize']}
  hyperParams['tofix']=[]


  classifynames = ['classify','comparison'] #'('classify', 'M'), ('classify', 'B'),('comparison', 'B'), ('comparison', 'M')]
  predictnames = ['predict','predictH'] #('predict', 'B'),('predict', 'M'), ('predictH', 'M'), ('predictH', 'B')]
  compositionnames =  ['composition']#('composition', '  # X#', '(#X#)', 'I', 'M'), ('composition', '#X#', '(#X#)', 'I', 'B'),('composition','#X#', '(#X#, #X#, #X#)', 'I', 'B'),('composition', '#X#', '(#X#, #X#)', 'I', 'B'), ('composition', '#X#', '(#X#, #X#)', 'I', 'M'), ('composition', '#X#', '(#X#, #X#,  # X#)', 'I', 'M')]
  embnames = ['word']#'('word',)]

  theta = installTheta(args['pars'], seed = seed, d=args['word'],noComparison=args['noComp'],predictH=args['predictH'])

  fullEvaluation(theta, seed=seed, noComparison=args['noComp'], predictH=args['predictH'])
  sys.exit()

  compareData,predictData = data.getTBs(seed=seed,noComparison=args['noComp'],predictH=args['predictH'], sets=['train','heldout'])
  if args['optimizer']=='adagrad': optimizer = opt.Adagrad(theta)#, lambdaL2 = args['lambda'], lr=args['alpha'])
  elif args['optimizer'] == 'adam':  optimizer = opt.Adam(theta)
  elif args['optimizer'] == 'sgd':  optimizer = opt.SGD(theta)
  else:
    print 'not a valid choice for optimizer:',args['optimizer']
    sys.exit()

  datasets = [compareData, predictData]
  names = ['compare expressions', 'scalarprediction']
  hyperParamsCompare = copy.deepcopy(hyperParams)
  hyperParamsPredict = copy.deepcopy(hyperParams)


  if args['kind']=='a1': #train alternating, but only train embeddings/ composition function during comparison training
    hyperParamsCompare['tofix'] += compositionnames + embnames
    datasets=[predictData,compareData]
    hypers = [hyperParamsPredict, hyperParamsCompare]
    phases=[6,4] #number of epochs for either phase
    names = ['scalarprediction','compare expressions']
  elif args['kind'] == 'a2': #train alternating, but only train embeddings/ composition function during prediction training
    hyperParamsPredict['tofix'] += compositionnames + embnames
    datasets = [compareData,predictData]
    hypers = [hyperParamsCompare,hyperParamsPredict]
    phases = [6, 4]  # number of epochs for either phase
    names = ['compare expressions','scalarprediction']
  elif args['kind'] == 'a3': #train alternating, and train embeddings/ composition function during both training phases
      datasets = [predictData, compareData]
      hypers = [hyperParamsPredict, hyperParamsCompare]
      phases = [5, 5]  # number of epochs for either phase
      names = ['scalarprediction', 'compare expressions']
  elif args['kind']=='c': # train on comparison only
    datasets=[compareData]
    hypers = [hyperParamsCompare]
    phases=[5] #number of epochs per round
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

  tr.alternate(optimizer, args['outDir'], datasets, hypers, phases, n = args['nEpochs'], names = names, verbose = (2 if args['verbose'] else 1),seed=seed)
  #predictData['train'].evaluate(theta,'train',n=20, verbose=True)
  fullEvaluation(theta, seed=seed, noComparison=args['noComp'], predictH=args['predictH'])

def mybool(string):
  if string in ['F', 'f', 'false', 'False']: return False
  elif string in ['T', 't', 'true', 'True']: return True
  else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-s', '--seed', type=str, help='Random seed to be used', required=True)
  parser.add_argument('-k', '--kind', type=str, help='scalar prediction (sp) or comparison (c) or alternating (a1/a2/a3)', required=True)
  parser.add_argument('-o','--outDir', type=str, help='Output dir to store pickled theta', required=True)
  parser.add_argument('-p','--pars', type=str, default='', help='File with pickled theta', required=False)
  # network hyperparameters:
  parser.add_argument('-dwrd','--word', type=int, default = 2, help='Dimensionality of leaves (word nodes)', required=False)
  # training hyperparameters:
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, default = 0, help='Regularization parameter lambdaL2', required=False)
  parser.add_argument('-a','--alpha', type=float, default = 0, help='Learning rate parameter alpha', required=False)
  parser.add_argument('-opt','--optimizer', type=str, choices = ['sgd','adagrad','adam'], help='Optimizer to be used', required=True)
  parser.add_argument('-c','--cores', type=int, default=1,help='The number of parallel processes', required=False)
  parser.add_argument('-nc','--noComp', type=mybool, help='Whether the comparison layer is removed', required=True)
  parser.add_argument('-ph', '--predictH', type=mybool, help='Whether there is a hidden layer for scalar prediction', required=True)
  parser.add_argument('-v', '--verbose', type=mybool, default=False, help='Whether a lot of output is printed',
                      required=False)

  args = vars(parser.parse_args())

  main(args)

