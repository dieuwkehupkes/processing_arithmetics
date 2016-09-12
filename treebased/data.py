import random
import core.classifier as cl
import core.myRNN as myRNN
import sys
sys.path.insert(0, '../commonFiles')
import arithmetics

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



def getTBs(digits, operators, predict = False, comparison = True, split = 0.2):
  languages_train = {'L1': 3000, 'L2': 3000, 'L4': 3000, 'L6': 3000}
  languages_test = {'L3': 400, 'L5': 400, 'L7': 400}

  trainData = arithmetics.mathTreebank(languages_train, digits)
  testData = arithmetics.mathTreebank(languages_test, digits)

  if not predict:
    items = trainData.examples[:]
    random.shuffle(items)
    ttb = compareClassifyTB(items[:int(split*len(items))])
    htb = compareClassifyTB(items[int(split * len(items)):])
    testtb = compareClassifyTB(testData.examples)

  return ttb, htb, testtb

