from classifier import *
import sys
import numpy as np
import pickle
from nltk import Tree
import random
import os
import trainArtificialData as ad
from collections import defaultdict

def grayCode(n):
  grays=[[0.0],[1.0]]
  while len(grays)<n+1:
    pGrays=grays[:]
    grays=[[0.0]+gray for gray in pGrays]+[[1.0]+gray for gray in pGrays[::-1]]
  return [np.array(gray) for gray in grays[1:]] # skip the first one: [0,0,0,0,0]




class TopNode():
  def __init__(self, nw,answer):
    self.root = nw
    self.answer = answer

  def activate(self,theta):
    self.root.root.forward(theta,activateIn = True, activateOut = False)

  def train(self, theta, gradient, activate=True, target = None, fixWords=False, fixWeights=False):
    if target is None: target = self.answer
    error = self.error(theta, target, activate)
    prediction = self.root.root.a
    true = theta[('word',)][target]

    delta = -(true-prediction)
    gradient[('word',)][target] -= delta
    delta = np.multiply(delta,self.root.root.ad)
    self.root.root.backprop(theta,delta,gradient,addOut=False,moveOn=True, fixWords = fixWords,fixWeights=fixWeights)
    return error


  def error(self,theta, target, activate=True):
    if target is None: target = self.answer
    if activate: self.activate(theta)
    prediction = self.root.root.a
    true = theta[('word',)][target]
    length = np.linalg.norm(true-prediction)
    return .5*length*length

  def evaluate(self,theta, target, sample=1):
    if target is None: target = self.answer
    return self.error(theta,target,True)

  def __str__(self):
    return 'TopNode: '+str(self.root)

class mathTreebank():
  def __init__(self, operators, digits, n=1000, lengths=range(1,6)):
    self.lengths = lengths
    self.n=n
    self.operators = operators
    self.grammar = {operator:{'(digit, '+operator+', digit)':5} for operator in self.operators+['is']}
    self.digits = [str(i) for i in digits]
    self.voc= self.digits[:]
#    self.voc.append('is')
    self.voc.append('UNKNOWN')
    self.voc.extend(self.operators)

  def getVoc(self):
    return self.voc

  def getGrammar(self):
    return self.grammar

  def getExamples(self, n=0, operators = [],digits=[]):
    if n == 0 : n =self.n
    if operators == []: operators = self.operators
    if digits == []: digits = self.digits
    examples = []
    while len(examples)<n:
      l =random.choice(self.lengths)
      tree = mathExpression(l,operators, digits)
      answer = tree.solve()
      if answer is None: continue
      if str(answer) not in self.digits: continue
      examples.append((tree,answer))
    return examples

class trainRNNTB():
  def __init__(self, mathTB):
    self.mathTB = mathTB
    self.n=mathTB.n
    self.labels = [None]
  def getExamples(self,n=0):
    if n==0: n = self.n
    examples = []
    for tree,answer in self.mathTB.getExamples(n):
      nw = myRNN.RNN(tree)
      classifier = TopNode(nw,str(answer))
      examples.append((classifier))
    return examples

class trainIORNNTB():
  def __init__(self, mathTB):
    self.mathTB = mathTB
    self.n=mathTB.n
    self.labels = [None]
  def getExamples(self,n=0, operators = [],digits=[]):
    if n==0: n = self.n
    examples = []
    for t,answer in self.mathTB.getExamples(n, operators, digits):
      tree = Tree('is',[t,Tree('operator',['is']), Tree('digit',[str(answer)])])
      nw = myIORNN.IORNN(tree)
      examples.append((nw,None))
#    print examples
    return examples


class mathExpression(Tree):
  def __init__(self,length, operators, digits):
    if length < 1: print 'whatup?'
    if length == 1:
      Tree.__init__(self,'digit',[random.choice(digits)])
    else:
      left = random.randint(1, length-1)
      right = length - left
      children = [mathExpression(l,operators, digits) for l in [left,right]]
      operator = random.choice(operators)
      children.insert(1,Tree('operator',[operator]))
      Tree.__init__(self,operator,children)
  def solve(self):
    if self.height()==2:
      try: return int(self[0])
      except DeprecationWarning:
        print self
        sys.exit()
    else:
      children = [c.solve() for c in [self[0],self[2]]]
      if None in children: return None
      operator = self.label()
      if operator== 'plus':
        return children[0]+children[1]
      elif operator== 'minus':
        return children[0]-children[1]
      elif operator== 'times':
        return children[0]*children[1]
      elif operator== 'div':
        try: return children[0]//children[1] # floor division
        except: return None
      elif operator== 'modulo':
        return children[0]%children[1]
      else:
        raise Exception('Cannot deal with operator '+str(operator))


class mathSentence(Tree):
  def __init__(self,length,operators, digits):
    answer = [None]
    while answer[0] not in voc:
      tree =mathExpression(length,operators, even)
      answer =Tree('digit',[str(tree.solve())])
    Tree.__init__(self,'is', [tree,Tree('operator',['is']),answer])

  def getTreeAndAnswer(self):
    return self[0],self[2]

  def corrupt(self,operators,digits):
    answer = self[1][0]
    candidate = answer
    while candidate==answer:
      candidate = random.choice(voc)
    self[1][0] = candidate
  def uncorrupt(self):
    self[1][0]=str(self[0].solve())

class resultClassifyTB():
  def __init__(self, mathTB):
    self.mathTB = mathTB
    self.labels = mathTB.digits
    self.n=mathTB.n

  def getExamples(self,n=0):
    if n==0: n = self.n
    examples = []
    for tree,answer in self.mathTB.getExamples(n):
      nw = myRNN.RNN(tree)
      classifier = Classifier([nw.root], self.labels, False)
      examples.append((classifier,str(answer)))
    return examples

class compareClassifyTB():
  def __init__(self, mathTB):
    self.mathTB = mathTB
    self.labels = ['<','=','>']
    self.n=mathTB.n

  def getExamples(self,n=0):
    if n == 0: n = self.n
    print n
    examples = []
    x = iter(self.mathTB.getExamples(2*n))

    while True:
      try:
        left, la = x.next()
        right, ra = x.next()
      except: return examples
      if la < ra: label = '<'
      elif la > ra: label = '>'
      else: label = '='
      classifier = Classifier([myRNN.RNN(left).root,myRNN.RNN(right).root], self.labels, False)
      examples.append((classifier,label))
    return examples

def install(thetaFile, kind='RNN', d=0):

  operators  = ['plus','minus']#,'times','div']#,'modulo]
  digits = [str(i) for i in range(-10,11)]
  tb = mathTreebank(operators, digits, n=5000, lengths = [1,2,4,6])
  tb2= mathTreebank(operators, digits, n=50, lengths = [3,5,7])
  print 'dimensionality:', d
  initWordsBin = d ==0
  if initWordsBin: print 'initialize words with Gray code'
  allData = defaultdict(dict)
  print  'load theta..', thetaFile
  try:
    with open(thetaFile, 'rb') as f:
      theta = pickle.load(f)
  except:
    voc= digits[:]
    voc.extend(operators)
    voc.append('is')
    voc.append('UNKNOWN')
    if initWordsBin:
      embeddings = grayCode(len(digits))
      d = len(embeddings[0])
      embeddings.extend([np.random.random_sample(d)*.2-.1 for op in operators+['is','UNKNOWN']])
    else: embeddings = None

    dims = {'inside':d,'outside':d,'word':d, 'maxArity':3}
    theta = myTheta.Theta(kind, dims, tb.grammar, embeddings, voc)
    theta.specializeHeads()
  print kind
  if kind == 'RNN':
  #  ttb =     trainRNNTB(tb)
  #  dtb =     trainRNNTB(mathTreebank(operators, digits, 1000,range(1,5)))

  #  ttb = resultClassifyTB(tb)
  #  dtb = resultClassifyTB(mathTreebank(operators, digits, 1000,range(1,8)))

    ttb = compareClassifyTB(tb)
    dtb = compareClassifyTB(tb2)
    nChildren =2
    theta.extend4Classify(nChildren, len(ttb.labels),dComparison = 0)

  else:
    ttb = trainIORNNTB(tb)
    dtb = trainIORNNTB(tb2)

  return theta, ttb, dtb
