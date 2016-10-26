from __future__ import division
from NN import *


class RNN():
    @classmethod
    def fromME(cls, me, activation):
        if me.label() == 'dummy':
            lhs = me.label()
            rhs = '(' + ', '.join([child.label() for child in me]) + ')'
            children = [cls.fromME(c, activation) for c in me]
            return Node(children, [], ('composition', lhs, rhs, 'I'), activation)
        else:
            return Leaf([], ('word',), key=me.label(), nonlinearity='identity')

    @classmethod
    def nodeLength(cls, node):
        if isinstance(node, Leaf):
            return 1
        else:
            return sum([cls.nodeLength(n) for n in node.inputs])

    @classmethod
    def getLeaves(cls, node):
        if isinstance(node, Leaf):
            return [node]
        else:
            leaves = []
            for n in node.inputs:
                leaves.extend(cls.getLeaves(n))
            return leaves

    def __init__(self, me, activation='tanh'):
        self.root = self.fromME(me, activation)
        self.length = me.length

    def activate(self, theta):
        self.root.forward(theta, True, False)
        return self.root.a

    def length(self):
        try:
            return self.length
        except:
            self.length = nodeLength(self.root)
            return self.length

    def leaves(self):
        return self.getLeaves(self.root)

    def maxArity(self, node=None):
        if node is None: node = self
        ars = [self.maxArity(c) for c in node.inputs]
        ars.append(len(node.inputs))
        return max(ars)

    def __str__(self):
        return str(self.root)
