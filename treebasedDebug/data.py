import random
import core.classifier as cl
import core.myRNN as myRNN
import sys
sys.path.insert(0, '../commonFiles')
import arithmetics
from collections import defaultdict, Counter


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
  def __init__(self, examples,comparison=False):
    self.labels = ['<','=','>']
    self.comparison = comparison
    self.examples = self.convertExamples(examples)


  def convertExamples(self,items):
    examples = []

    for left, right, label in items:
      if not self.comparison: classifier = cl.ClassifierNoComparison([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, False)
      else: classifier = cl.Classifier([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, False)
      examples.append((classifier,label))
    return examples

  def evaluate(self, theta, name='', n=0, verbose=1):
    if n == 0: n = len(self.examples)
    error = 0.0
    true = 0.0
    confusion = defaultdict(Counter)
    for nw, target in self.getExamples(n):
      error += nw.evaluate(theta, target)
      prediction = nw.predict(theta, None, False, False)
      confusion[target][prediction] += 1
      if prediction == target:
        true += 1
      else:
        if verbose == 2: print 'wrong prediction:', prediction, 'target:', target
    accuracy = true / n
    loss = error / n
    if verbose == 1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
    if verbose == 1: print '\tLoss:', loss, 'Accuracy:', accuracy, 'Confusion:'
    if verbose == 1: print confusionS(confusion, self.labels)
    return loss, accuracy


class ScalarPredictionTB(TB):
  def __init__(self, examples):
    self.examples = self.convertExamples(examples)
  def convertExamples(self,items):
    examples = []
    for tree,label in items:
      predictor = cl.Predictor(myRNN.RNN(tree).root)
      examples.append((predictor,label))
    return examples

  def evaluate(self, theta, name='', n=0, verbose=1):
    if n == 0: n = len(self.examples)
    sse = 0.0
    sspe = defaultdict(float)  # .0
    lens = defaultdict(int)  # .0
    true = 0.0
    for nw, target in self.getExamples(n):

      pred = nw.predict(theta, roundoff=True)
      sse += nw.evaluate(theta, target, activate=False, roundoff=False)
      length = nw.length
      sspe[length] += nw.error(theta, target, activate=False, roundoff=True)
      lens[length] += 1
      if target == pred: true += 1
      if verbose == 2:
        length = (len(str(nw).split(' ')) + 3) / 4
        print 'length:', length, (
        'right' if target == pred else 'wrong'), 'prediction:', pred, 'target:', target, 'error:', nw.error(theta,
                                                                                                            target,
                                                                                                            activate=False,
                                                                                                            roundoff=True), '(' + str(
          nw.error(theta, target, activate=False, roundoff=False)) + ')'

    mse = sse / n
    accuracy = true / n
    mspe = sum(sspe.values()) / n
    if verbose == 1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
    if verbose == 1: print '\tLoss (MSE):', mse, 'Accuracy:', accuracy, 'MSPE:', mspe
    if verbose == 1: print '\tMSPE per length: ', [
      (length, (sspe[length] / lens[length] if lens[length] > 0 else 'undefined')) for length in sspe.keys()]
    return mse, accuracy, mspe


# def getTBs(digits, operators, predict = False, comparison = False, split = 0.1):
#   languages_train = {'L1': 10000, 'L2': 10000, 'L4': 10000, 'L6': 10000}
#   languages_test = {'L3': 400, 'L5': 400, 'L7': 400}
#
#   trainData = arithmetics.mathTreebank(languages_train, digits)
#   testData = arithmetics.mathTreebank(languages_test, digits)
#
#   if not predict:
#     items = trainData.examples[:]
#     random.shuffle(items)
#     htb = CompareClassifyTB(items[:int(split*len(items))],comparison=comparison)
#     ttb = CompareClassifyTB(items[int(split * len(items)):], comparison=comparison)
#     testtb = CompareClassifyTB(testData.examples, comparison=comparison)
#     compareData={'train':ttb,'heldout':htb,'test':testtb}
#
#     htb = ScalarPredictionTB(items[:int(split * len(items))])
#     ttb = ScalarPredictionTB(items[int(split * len(items)):])
#     testtb = ScalarPredictionTB(testData.examples)
#     predictData = {'train':ttb,'heldout':htb,'test':testtb}
#
#   return compareData, predictData

def getComparisonTest(seed, comparison):
  for name, mtb in arithmetics.test_treebank(seed):
    yield name, CompareClassifyTB(mtb.pairedExamples, comparison=comparison)


def getComparisonTBs(seed, comparison):
  data = {}
  for kind in 'train','heldout':
    if kind == 'train': mtb = arithmetics.training_treebank(seed)
    elif kind == 'heldout': mtb = arithmetics.heldout_treebank(seed)
    print len(mtb.pairedExamples), len(mtb.examples)
    data[kind] = CompareClassifyTB(mtb.pairedExamples, comparison=comparison)
  return data


