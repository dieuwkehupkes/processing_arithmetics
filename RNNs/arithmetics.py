import sys
import numpy as np
import pickle
from nltk import Tree
import random
import os
from collections import defaultdict


class mathTreebank():
    def __init__(self):
        self.examples = []          # attribute containing examples of the treebank
        self.operators = set([])    # attribute containing operators in the treebank
        self.digits = set([])       # digits in the treebank

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
        digits = [str(i) for i in digits]
        self.digits = self.digits.union(set(digits))
        self.operators = self.operators.union(set(operators))
        while len(examples) < n:
            l = random.choice(lengths)
            tree = mathExpression(l, operators, digits, branching=branching)
            answer = tree.solve()
            if answer is None:
                continue
            if not (min <= answer <= max):
                continue
            examples.append((tree,answer))
        return examples

    def addExamples(self, operators=['+','-'], digits=np.arange(-19,19), branching=None, min=-60, max=60, n=1000, lengths=range(1,6)):
        """
        Add examples to treebank.
        """
        self.examples += self.generateExamples(operators=operators, digits=digits, branching=branching, min=min, max=max, n=n, lengths=lengths)

    def write_to_file(self, filename):
        """
        Generate a file containing the treebank.
        Every tree element is separated by spaces, a tab
        separates the answer from the sentence. E.g
        ( ( 5 + 6 ) - 3 )   8
        """
        f = open(filename, 'wb')
        for example in self.examples:
            f.write(str(example[0])+'\t'+str(example[1])+'\n')
        f.close()


class mathExpression(Tree):
    def __init__(self,length, operators, digits, branching=None):
        if length < 1: print 'whatup?'
        if length == 1:
            try:
                Tree.__init__(self,'digit',[random.choice(digits)])
            except IndexError:
                Tree.__init__(self, 'operator', [random.choice(operators)])
        else:
            if branching == 'left':
                left, right = length-1, 1
            elif branching == 'right':
                left, right = 1, length-1
            else:
                left = random.randint(1, length-1)
                right = length - left
            children = [mathExpression(l,operators, digits, branching) for l in [left,right]]
            operator = random.choice(operators)
            children.insert(1, mathExpression(1, [operator], []))
            Tree.__init__(self,operator,children)

    def solve(self):
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

digits = [str(i) for i in np.arange(-5,5)]
ops = ['+','-']
m = mathTreebank()
m.addExamples(n=5, lengths=[5], branching='left')
m.write_to_file('treebank')
