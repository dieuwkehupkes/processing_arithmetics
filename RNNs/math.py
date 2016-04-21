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
    def __init__(self):
        self.examples = []      # create attribute containing examples of the treebank

    def generateExamples(self, operators, digits, branching=None, min=-60, max=60, n=1000, lengths=range(1,6)):
        """
        :param operators:       operators to be used in
                                arithmetic expressions \in {+,-,\,*}
        :param digits:          range with digits (list)
        :param branching:       set to 'left' or 'right' to restrict branching of trees
        :param n:               number of sentences in tree bank
        :param min:             min outcome of the composition function
        :param max:             max outcome of the composition function
        :param lengths:         number of numeric leaves of expressions
        """
        examples = []
        while len(examples) < n:
            l = random.choice(self.lengths)
            tree = mathExpression(l, operators, digits, branching=branching)
            answer = tree.solve()
            if answer is None: continue
            if str(answer) not in self.digits: continue
            examples.append((tree,answer))
        return examples

    def addExamples(self, operators=['+','-'], digits=np.arange(-19,19), branching=None, min=-60, max=60, n=1000, lengths=range(1,6)):
        """
        Add examples to treebank.
        """
        self.examples.append(self.generateExamples(operators=operators, digits=digits, branching=branching, min=min, max=max, n=n, lengths=lengths))

    def __str__(self):
        """
        """
        raise NotImplementedError("implement string function")


class mathExpression(Tree):
    def __init__(self,length, operators, digits, branching=None):
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

    def solve2(self):
        """
        Evaluate the expression
        """
        return eval(self.__str__())

    def __str__(self):
        """
        Return string representation of tree.
        """
        if len(self) > 1:
            return '( '+' '.join([str(child) for child in self])+' )'
        else:
          return self[0]


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
