import random
import core.classifier as cl
import core.myRNN as myRNN
import sys
sys.path.insert(0, '../commonFiles')
import arithmetics
from collections import defaultdict, Counter


def confusionS(matrix,labels):
  if True:#len(labels)<15:
    s = ''
    for label in labels:
      s+='\t'+label
    s+='\n'
    for t in labels:
      s+= t
      for p in labels:
        s+= '\t'+str(matrix[t][p])
      s+='\n'

  else: #compacter representations
    s = 'target: (prediction,times)\n'
    for t,ps in matrix.items():
      s+=str(t)+':'
      for p, v in ps.items():
        s+= ' ('+p+','+str(matrix[t][p])+')'
      s+='\n'
  return s

class compareClassifyTB():
  def __init__(self, examples,noComparison=False):
    self.labels = ['<','=','>']
    self.noComp = noComparison
    self.examples = self.convertExamples(examples)

  def getExamples(self,n=0):
    if n==0: n = len(self.examples)
    random.shuffle(self.examples)
    return self.examples[:n]

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
      print 'Evaluation on ' + name + ' data (' + str(n) + ' examples):'
      print 'Loss:', loss, 'Accuracy:', accuracy, 'Confusion:'
      print confusionS(confusion, self.labels)


def getTBs(digits, operators, predict = False, noComparison = False, split = 0.1):
  languages_train = {'L1': 3000, 'L2': 3000, 'L4': 3000, 'L6': 3000}
  languages_test = {'L3': 400, 'L5': 400, 'L7': 400}

  trainData = arithmetics.mathTreebank(languages_train, digits)
  testData = arithmetics.mathTreebank(languages_test, digits)

  if not predict:
    items = trainData.examples[:]
    random.shuffle(items)
    htb = compareClassifyTB(items[:int(split*len(items))],noComparison=noComparison)
    ttb = compareClassifyTB(items[int(split * len(items)):], noComparison=noComparison)
    testtb = compareClassifyTB(testData.examples)

  return ttb, htb, testtb

