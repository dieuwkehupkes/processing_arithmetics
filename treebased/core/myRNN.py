from __future__ import division
from NN import *
import random
import sys
import numpy as np


def this2RNN(nltkTree, activation):
  if nltkTree.height()>2:
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    children = [this2RNN(t, activation) for t in nltkTree]
    rnn = Node(children, [], ('composition',lhs,rhs,'I'), activation)
  else:
    rnn = Leaf([],('word',), key=nltkTree[0],nonlinearity='identity')
  return rnn

def nodeLength(node):
  if isinstance(node,Leaf): return 1
  else: return sum([nodeLength(n) for n in node.inputs])
def getLeaves(node):
#  print node
  if isinstance(node,Leaf): return [node]
  else:
    leaves =[]
    for n in node.inputs:
      leaves.extend(getLeaves(n))
    return leaves


class RNN():
  def __init__(self, nltkTree, activation='tanh'):
    self.root =this2RNN(nltkTree, activation)
    self.length = len(nltkTree.leaves())
  def activate(self,theta):
    self.root.forward(theta, True, False)
    return self.root.a

  def length(self):
    try: return self.length
    except:
      self.length= nodeLength(self.root)
      return self.length

  def leaves(self):
    return getLeaves(self.root)

  def maxArity(self,node=None):
    if node is None: node = self
    ars = [self.maxArity(c) for c in node.inputs]
    ars.append(len(node.inputs))
    return max(ars)
  def __str__(self):
    return str(self.root)