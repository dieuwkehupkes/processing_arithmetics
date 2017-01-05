from __future__ import division
import activation
import numpy as np

'''
 These classes were written aimed at the Recursive Neural Network, where a layer has multiple input layers.
 In forward mode, the concatenation of representations of the input layers
                   is used as input to compute activation of a layer.
 In backward mode, the delta message is split and distributed over the input layers.

A Node (which is basically a Neural Network Layer) has
- inputs: list of Nodes (that it will concatenate to serve as input)
- cat: the key (tuple of strings) to obtain matrix and bias from theta
- nonlinearity: name of the activation function
'''
class Node():
    def __init__(self, inputs, cat, nonlinearity):
        self.inputs = inputs
        self.cat = cat
        self.nonlin = nonlinearity

    def forward(self, theta, activateIn=True):
        if activateIn:
            [i.forward(theta, activateIn) for i in self.inputs]
        self.inputsignal = np.concatenate([c.a for c in self.inputs])
        self.dinputsignal = np.concatenate([c.ad for c in self.inputs])

        M = theta[self.cat + ('M',)]
        b = theta[self.cat + ('B',)]
        if M is None or b is None: raise RuntimeError(
            'Fail to forward node, no matrix and bias vector:' + str(self.cat))
        self.z = M.dot(self.inputsignal) + b
        self.a, self.ad = activation.activate(self.z, self.nonlin)


    def backprop(self, theta, delta, gradient, moveOn=True):
        M = theta[self.cat + ('M',)]
        gradient[self.cat + ('M',)] += np.outer(delta, self.inputsignal)
        gradient[self.cat + ('B',)] += delta

        deltaB = np.multiply(np.transpose(M).dot(delta), self.dinputsignal)
        if moveOn:
            lens = [len(c.a) for c in self.inputs]
            splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
            [inputNode.backprop(theta, delt, gradient, moveOn) for inputNode, delt in zip(self.inputs, np.split(deltaB, splitter))]
        else:
            return deltaB

    def __str__(self):
        return '( ' + ' '.join([str(child) for child in self.inputs]) + ' )'

'''
A Leaf is a special kind of Node with no inputs. It has a 'key', which is the word it represents
'''
class Leaf(Node):
    def __init__(self, cat, key, nonlinearity='identity'):
        Node.__init__(self, inputs=[], cat=cat, nonlinearity= nonlinearity)
        self.key = key

    def forward(self, theta, activateIn=True):
        self.z = theta[self.cat][self.key]
        self.a, self.ad = activation.activate(self.z, self.nonlin)

    def backprop(self, theta, delta, gradient, moveOn=False):
        gradient[self.cat][self.key] += delta

    def aLen(self, theta):
        return len(theta[self.cat][self.key])

    def __str__(self):
        return str(self.key)


class RNN():
    @classmethod #make an RNN from a mathExpression
    def fromME(cls, me, activation):
        if me.label() == 'dummy':
            lhs = me.label()
            rhs = '(' + ', '.join([child.label() for child in me]) + ')'
            children = [cls.fromME(c, activation) for c in me]
            return Node(children, ('composition', lhs, rhs, 'I'), activation)
        else:
            return Leaf(cat=('word',), key=me.label(), nonlinearity='identity')

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
        self.root.forward(theta, True)
        return self.root.a

    def length(self):
        try:
            return self.length
        except:
            self.length = self.nodeLength(self.root)
            return self.length

    def leaves(self):
        return self.getLeaves(self.root)

    def maxArity(self, node=None):
        if node is None: node = self.root
        ars = [self.maxArity(c) for c in node.inputs]
        ars.append(len(node.inputs))
        return max(ars)

    def __str__(self):
        return str(self.root)

class Predictor(Node):
    def __init__(self, child):
        Node.__init__(self, [child.root], [], ('predict',), 'identity')
        self.length = child.length

    def predict(self, theta, activate=True, roundoff=True, verbose=False):
        if activate: self.forward(theta)
        if roundoff:
            return int(round(self.a[0], 0))
        else:
            return self.a[0]

    def error(self, theta, target, activate=True, roundoff=False):
        if activate: self.forward(theta)
        pred = self.predict(theta, activate=activate, roundoff=roundoff)
        return (target - pred) ** 2

    def train(self, theta, gradient, activate, target):
        if activate: self.forward(theta)
        delta = -2 * (target - self.a)
        # TODO: check delta message
        self.backprop(theta, delta, gradient, moveOn=True)

        return self.error(theta, target, False)

    def evaluate(self, theta, target, activate=True, roundoff=False):
        return self.error(theta, target, activate, roundoff=roundoff)


class Classifier(Node):
    '''
    Comparison: whether a hidden 'comparison' layer should be added
    '''
    def __init__(self, children, labels, comparison):
        self.comparison = comparison
        if comparison:
            comparisonlayer = Node(children, [self], ('comparison',), 'ReLU')
            Node.__init__(self, [comparisonlayer], [], ('classify',), 'softmax')
        else:
            Node.__init__(self, inputs=children,cat=('classify',), nonlinearity='softmax')
        self.labels = labels

    def train(self, theta, gradient, activate, target):
        if activate: self.forward(theta)
        delta = np.copy(self.a)
        true = self.labels.index(target)
        delta[true] -= 1
        self.backprop(theta, delta, gradient, moveOn=True)
        error = self.error(theta, target, False)
        return error

    def error(self, theta, target, activate=True):
        if activate: self.forward(theta)

        try:
            err = -np.log(self.a[self.labels.index(target)])
        except:
            err = -np.log(1e-10)
        return err

    def evaluate(self, theta, target, activate=True):
        if activate: self.forward(theta)
        return self.error(theta, target)

    def predict(self, theta, activate=True):
        if activate: self.forward(theta)
        return self.labels[self.a.argmax(axis=0)]

    def __str__(self):
        if self.comparison:
            return 'classify: ' + ', '.join([str(ch) for ch in self.inputs[0].inputs])
        else:
            return 'classify: ' + ', '.join([str(ch) for ch in self.inputs])
