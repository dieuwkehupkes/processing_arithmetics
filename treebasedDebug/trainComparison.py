from __future__ import division
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

import data
import pickle
import core.myTheta as myTheta
import core.trainingRoutines as tr
from core import Optimizer
import argparse
import sys, os
import core.gradient_check as check

def installTheta(thetaFile, seed, d, comparison):
  if thetaFile != '':
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
    print 'initialized theta from file:', thetaFile
    if not comparison and ('comparison','M') in theta.keys():
      theta.extend4Classify(2, 3, -1)
    if comparison and ('comparison', 'M') not in theta.keys():
      theta.extend4Classify(2, 3, 3 * d)

    if '+' not in theta[('word',)].voc:
      theta[('word',)].extendVocabulary(['+','-'])
      theta[('word',)]['+']=theta[('word',)]['plus']
      theta[('word',)]['-'] = theta[('word',)]['minus']

  else:
    print 'initializing theta from scratch'
    dims = {'inside': d, 'outside': d, 'word': d, 'minArity': 3, 'maxArity': 3}
    voc = ['UNKNOWN'] + [str(w) for w in data.arithmetics.ds] + data.arithmetics.ops
    theta = myTheta.Theta('RNN', dims, embeddings=None, vocabulary=voc, seed = seed)
    if not comparison: theta.extend4Classify(2,3,-1)
    else: theta.extend4Classify(2,3,3*d)
  theta.extend4Prediction(-1)
  return theta


def main(args):
  print args
  hyperParams={k:args[k] for k in ['bSize']}
  hyperParams['toFix']=[]
  hyperParamsCompare = hyperParams.copy()
  hyperParamsPredict = hyperParams.copy()

  theta = installTheta(args['pars'],seed=args['seed'],d=args['word'],comparison=args['comparison'])

  kind = args['kind']

  if kind =='c':
    hypers = [hyperParamsCompare]
    names = ['comparison']
  elif kind =='p':
    hypers = [hyperParamsPredict]
    names = ['prediction']
  else: sys.exit()

  datasets = [data.getTBs(seed=args['seed'], kind = name, comparison=args['comparison']) for name in names]

  phases = [2]  # number of epochs for either phase

  opt = args['optimizer']
  if opt =='adagrad': optimizer = Optimizer.Adagrad(theta,lr = args['learningRate'])
  elif opt =='adam': optimizer = Optimizer.Adam(theta,lr = args['learningRate'])
  elif opt =='sgd': optimizer = Optimizer.SGD(theta,lr = args['learningRate'])

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



  #tr.alternate(theta, datasets,alt=phases, outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs']//10, names=names)
  evals = tr.alternate(optimizer, datasets, alt=phases,outDir=args['outDir'], hyperParams=hypers, n=args['nEpochs']//2, names=names)

  for name, eval in evals.items():
    toplot = [e[key] for e in eval for key in e  if 'loss' in key]
    #zip(eval)
    #loss = eval[0][1]
    plt.plot(xrange(len(toplot)), toplot,label=name)
  plt.legend()
  plt.title([ key for key in eval[0].keys() if 'loss'  in key][0])
  plt.savefig(os.path.join(args['outDir'],'convergencePlot.png'))
  #plt.show()

  #predictData['train'].evaluate(theta,'train',n=20, verbose=True)

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
  parser.add_argument('-c','--comparison', type=mybool, default=False, help='Whether a hidden (comparison) layer is used', required=False)
  parser.add_argument('-dwrd','--word', type=int, default = 2, help='Dimensionality of leaves (word nodes)', required=False)
  # training hyperparameters:
  parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adagrad', 'adam'], help='Activation of hidden layer', required=False)
  parser.add_argument('-s', '--seed', type=int, help='Random seed to be used', required=True)
  parser.add_argument('-n','--nEpochs', type=int, help='Maximal number of epochs to train per phase', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 50, help='Batch size for minibatch training', required=False)
  parser.add_argument('-l','--lambda', type=float, help='Regularization parameter lambdaL2', required=True)
  parser.add_argument('-lr','--learningRate', type=float, help='Learning rate parameter', required=True)


  args = vars(parser.parse_args())

  main(args)

