import random
import core.classifier as cl
import core.myRNN as myRNN
import sys
sys.path.insert(0, '../commonFiles')
import arithmetics
from collections import defaultdict, Counter
import re

digits = [str(d) for d in arithmetics.ds]
operators = arithmetics.ops

class TB():
  def __init__(self, examples):
    self.examples = examples

  def getExamples(self, n=0):
    if n == 0: n = len(self.examples)
    random.shuffle(self.examples)
    return self.examples[:n]


def confusionS(matrix,labels):
  if len(labels)<15:
    s = '\t'
    for label in labels:
      s+='\t'+label
    s+='\n\t'
    for t in labels:
      s+= t
      for p in labels:
        s+= '\t'+str(matrix[t][p])
      s+='\n\t'

  else: #compacter representations
    s = 'target: (prediction,times)\n\t'
    for t,ps in matrix.items():
      s+=str(t)+':'
      for p, v in ps.items():
        s+= ' ('+p+','+str(matrix[t][p])+')'
      s+='\n\t'
  return s

class CompareClassifyTB(TB):
  def __init__(self, examples,noComparison=False):
    self.labels = ['<','=','>']
    self.noComp = noComparison
    self.examples = self.convertExamples(examples)


  def convertExamples(self,items):
    examples = []
    x = iter(items)
    while True:
      try:
        left, la = x.next()
        right, ra = x.next()
      except: break#return examples
      if la < ra: label = '<'
      elif la > ra: label = '>'
      else: label = '='
      classifier = cl.Classifier([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, noComparison=self.noComp)
      examples.append((classifier,label))
    return examples

  def evaluate(self, theta, name='', n=0, verbose=1):
      if n == 0: n = len(self.examples)
      error = 0.0
      true = 0.0
      confusion = defaultdict(Counter)
      for nw, target in self.getExamples(n):
        error += nw.evaluate(theta, target)
        prediction = nw.predict(theta, False, False)
        confusion[target][prediction] += 1
        if prediction == target: true += 1
        else:
          if verbose==2: print 'wrong prediction:', prediction,'target:',target
      accuracy = true / n
      loss = error / n
      if verbose ==1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
      if verbose == 1: print '\tLoss:', loss, 'Accuracy:', accuracy, 'Confusion:'
      if verbose == 1: print confusionS(confusion, self.labels)
      return loss, accuracy

class ScalarPredictionTB(TB):
  def __init__(self, examples, hiddenLayer = False):
    self.examples = self.convertExamples(examples, hiddenLayer)
  def convertExamples(self,items, hiddenLayer):
    examples = []
    for tree,label in items:
      predictor = cl.Predictor(myRNN.RNN(tree).root, hiddenLayer=hiddenLayer)
      examples.append((predictor,label))
    return examples

  def evaluate(self, theta, name='', n=0, verbose = 1):
    if n == 0: n = len(self.examples)
    sse = 0.0
    sspe = defaultdict(float)#.0
    lens = defaultdict(int)  # .0
    true = 0.0
    for nw, target in self.getExamples(n):

      pred = nw.predict(theta,roundoff=True)
      sse += nw.evaluate(theta, target, activate=False, roundoff = False)
      length = nw.length
      sspe[length] += nw.error(theta, target, activate=False, roundoff = True)
      lens[length]+=1
      if target==pred: true +=1
      if verbose==2:
        length = (len(str(nw).split(' '))+3)/4
        print 'length:',length,('right' if target==pred else 'wrong'), 'prediction:' , pred, 'target:', target, 'error:', nw.error(theta, target, activate=False, roundoff = True),'('+str(nw.error(theta, target, activate=False, roundoff = False))+')'

    mse=sse/n
    accuracy = true / n
    mspe = sum(sspe.values()) / n
    if verbose == 1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
    if verbose == 1: print '\tLoss (MSE):', mse, 'Accuracy:', accuracy, 'MSPE:', mspe
    if verbose == 1: print '\tMSPE per length: ', [(length,(sspe[length]/lens[length] if lens[length]>0 else 'undefined')) for length in sspe.keys()]
    return mse, accuracy, mspe

def getTestTBs(seed,noComparison, predictH,subset = 0):
  for lan, mathtb in arithmetics.test_treebank(seed):
    items = mathtb.examples
    if subset > 0: items = items[:subset]
    yield lan, CompareClassifyTB(items, noComparison=noComparison), ScalarPredictionTB(items, hiddenLayer=predictH)

def getTBs(seed, noComparison, predictH, sets=['train','heldout'], subset = 0):
  predictData={}
  compareData={}
  for name in sets:
    if name=='train': mathtb = arithmetics.training_treebank(seed)
    elif name =='heldout': mathtb = arithmetics.heldout_treebank(seed)
    items = mathtb.examples
    if subset > 0: items = items[:subset]
    predictData[name] = ScalarPredictionTB(items, hiddenLayer=predictH)
    compareData[name] = CompareClassifyTB(items, noComparison=noComparison)

  return compareData, predictData
#
# def makeFile(digits, theta, directory):
#   languages = [{'L1': 3000, 'L2': 3000, 'L4': 3000, 'L6': 3000}, {'L3': 400, 'L5': 400, 'L7': 400}]
#   #print 'Training languages:', str(languages_train)
#   #print 'Testing languages:', str(languages_test)
#   import pickle, numpy as np
#
#   train = True
#   for i in range(2):
#     tb = arithmetics.mathTreebank(languages[i], digits)
#     inputs = []
#     outputs = []
#     strings = []
#     for me, answer in tb.examples:
#       nw = myRNN.RNN(me)
#       nw.activate(theta)
#       strings.append(str(me))
#       inputs.append(nw.root.a)
#       outputs.append(answer)
#
#     for data, name in [(inputs,'X_'),(outputs,'Y_'),(strings,'strings_')]:
#       with open(directory + '/' +name + ('train' if train else 'test') + '.pkl', 'wb') as f:
#         pickle.dump(np.array(data),f)
#
#     with open(directory + '/' + ('train' if train else 'test') + 'Data.txt', 'w') as f:
#           f.write('inputs='+str(inputs)+'\n')
#           f.write('outputs=' + str(outputs)+'\n')
#           f.write('strings='+str(strings)+'\n')
#     train = False
#
#
#
