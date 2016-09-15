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
      if self.noComp: classifier = cl.ClassifierNoComparison([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, False)
      else: classifier = cl.Classifier([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, False)
      examples.append((classifier,label))
    return examples

  def evaluate(self, theta, name='', n=0):
      if n == 0: n = len(self.examples)
      error = 0.0
      true = 0.0
      confusion = defaultdict(Counter)
      for nw, target in self.getExamples(n):
        error += nw.evaluate(theta, target)
        prediction = nw.predict(theta, None, False, False)
        confusion[target][prediction] += 1
        if prediction == target: true += 1
      accuracy = true / n
      loss = error / n
      print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
      print '\tLoss:', loss, 'Accuracy:', accuracy, 'Confusion:'
      print confusionS(confusion, self.labels)

class ScalarPredictionTB(TB):
  def __init__(self, examples):
    self.examples = self.convertExamples(examples)
  def convertExamples(self,items):
    examples = []
    for tree,label in items:
      predictor = cl.Predictor(myRNN.RNN(tree).root)
      examples.append((predictor,label))
    return examples

  def evaluate(self, theta, name='', n=0, verbose = False):
    if n == 0: n = len(self.examples)
    sse = 0.0
    sspe = 0.0
    true =0.
    for nw, target in self.getExamples(n):
      pred = nw.predict(theta,roundoff=True)
      sse += nw.error(theta, target, activate=False, roundoff = False)
      sspe += nw.error(theta, target, activate=False, roundoff = True)
      if target==pred: true +=1
      if verbose: print ('right' if target==pred else 'wrong'), 'prediction:' , pred, 'target:', target, 'error:', nw.error(theta, target, activate=False, roundoff = True),'('+str(nw.error(theta, target, activate=False, roundoff = False))+')'
    mse=sse/n
    accuracy = true / n
    mspe = sspe / n
    print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
    print '\tLoss (MSE):', mse, 'Accuracy:', accuracy, 'MSPE:', mspe


def getTBs(digits, operators, predict = False, noComparison = False, split = 0.1):
  languages_train = {'L1': 10000, 'L2': 10000, 'L4': 10000, 'L6': 10000}
  languages_test = {'L3': 400, 'L5': 400, 'L7': 400}

  trainData = arithmetics.mathTreebank(languages_train, digits)
  testData = arithmetics.mathTreebank(languages_test, digits)

  if not predict:
    items = trainData.examples[:]
    random.shuffle(items)
    htb = CompareClassifyTB(items[:int(split*len(items))],noComparison=noComparison)
    ttb = CompareClassifyTB(items[int(split * len(items)):], noComparison=noComparison)
    testtb = CompareClassifyTB(testData.examples, noComparison=noComparison)
    compareData={'train':ttb,'heldout':htb,'test':testtb}

    htb = ScalarPredictionTB(items[:int(split * len(items))])
    ttb = ScalarPredictionTB(items[int(split * len(items)):])
    testtb = ScalarPredictionTB(testData.examples)
    predictData = {'train':ttb,'heldout':htb,'test':testtb}

  return compareData, predictData

