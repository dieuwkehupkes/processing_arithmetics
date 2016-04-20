import sys
import numpy as np
import pickle
from nltk import Tree
import random
import os
from collections import defaultdict

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
        self.n = n
        self.operators = operators
        self.grammar = {operator:{'(digit, '+operator+', digit)':5} for operator in self.operators+['is']}
        self.digits = [str(i) for i in digits]

    def getVoc(self):
        return self.digits+self.operators+['UNKNOWN']       #,'is']

    def getGrammar(self):
        return self.grammar

    def getExamples(self, n=0, operators = [],digits=[]):
        if n == 0 : n = self.n
        if operators == []: operators = self.operators
        if digits == []: digits = self.digits
        examples = []
        while len(examples) < n:
            l = random.choice(self.lengths)
            tree = mathExpression(l,operators, digits)
            answer = tree.solve()
            if answer is None: continue
            if str(answer) not in self.digits: continue
            examples.append((tree,answer))
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
        if self.height() == 2:
            try: return int(self[0])
            except DeprecationWarning:
                print self
                sys.exit()
        else:
            children = [c.solve() for c in [self[0],self[2]]]
            if None in children: return None
            operator = self.label()
            if operator == '+':
                return children[0]+children[1]
            elif operator == '-':
                return children[0]-children[1]
            elif operator == 'times':
                return children[0]*children[1]
            elif operator == 'div':
                try: return children[0]//children[1]     # floor division
                except: return None
            elif operator == 'modulo':
                return children[0] % children[1]
            else:
                raise Exception('Cannot deal with operator '+str(operator))


"""
def install(thetaFile, kind='RNN', d=0):

    operators    = ['+','-']#,'*','/']#,'modulo]
    digits = [str(i) for i in range(-10,11)]
    tb = mathTreebank(operators, digits, n=5000, lengths = [1,2,4,6])
    tb2 = mathTreebank(operators, digits, n=50, lengths = [3,5,7])
    print 'dimensionality:', d
    initWordsBin = d ==0
    if initWordsBin: print 'initialize words with Gray code'
    print 'load theta..', thetaFile
"""
