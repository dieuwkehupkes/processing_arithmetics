from __future__ import division
import random
from .core import NN as NN
from .. import treebanks as arithmetics
from collections import defaultdict, Counter

class TB():
    def __init__(self, examples):
        self.examples = examples

    def getExamples(self, n=0):
        if n == 0: n = len(self.examples)
        random.shuffle(self.examples)
        return self.examples[:n]


def confusionS(matrix, labels):
    if len(labels) < 15:
        s = '\t'
        for label in labels:
            s += '\t' + label
        s += '\n\t'
        for t in labels:
            s += t
            for p in labels:
                s += '\t' + str(matrix[t][p])
            s += '\n\t'

    else:  # compacter representations
        s = 'target: (prediction,times)\n\t'
        for t, ps in matrix.items():
            s += str(t) + ':'
            for p, v in ps.items():
                s += ' (' + p + ',' + str(matrix[t][p]) + ')'
            s += '\n\t'
    return s


class RNNTB(TB):
    def __init__(self, examples):
        self.examples = [(NN.RNN(me), target) for (me, target) in examples]


class CompareClassifyTB(TB):
    def __init__(self, examples, comparison=False):
        self.labels = ['<', '=', '>']
        self.comparison = comparison
        self.examples = self.convertExamples(examples, comparison)

    def convertExamples(self, items, comparison):
        examples = []

        for left, right, label in items:
            classifier = NN.Classifier(children=[NN.RNN(left).root, NN.RNN(right).root], labels=self.labels,
                                       comparison=comparison)
            examples.append((classifier, label))
        return examples

    def evaluate(self, theta, name='', n=0, verbose=1):
        if n == 0: n = len(self.examples)
        error = 0.0
        true = 0.0
        diffs = []
        confusion = defaultdict(Counter)
        for nw, target in self.getExamples(n):
            error += nw.evaluate(theta, target)
            prediction = nw.predict(theta=theta, activate = False)
            confusion[target][prediction] += 1
            if prediction == target:
                true += 1
            else:
                string = str(nw)
                results = eval(string.split(':')[1])
                diffs.append(abs(results[0] - results[1]))
                if verbose == 2:      print 'wrong prediction:', prediction, 'target:', target, string, results, 'difference:', abs(
                    results[0] - results[1])

        accuracy = true / n
        loss = error / n
        if verbose == 1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
        if verbose == 1: print '\tLoss:', loss, 'Accuracy:', accuracy, 'Confusion:'
        if verbose == 1: print confusionS(confusion, self.labels)
        if verbose > 0: print '\taverage absolute difference of missclassified examples:', sum(diffs) / len(diffs)
        return {'loss (cross entropy)': loss, 'accuracy': accuracy}


class ScalarPredictionTB(TB):
    def __init__(self, examples):
        self.examples = self.convertExamples(examples)

    def convertExamples(self, items):
        examples = []
        for tree, label in items:
            predictor = NN.Predictor(NN.RNN(tree))
            examples.append((predictor, label))
        return examples

    def evaluate(self, theta, name='', n=0, verbose=1):
        if n == 0: n = len(self.examples)
        sse = 0.0
        sspe = defaultdict(float)
        lens = defaultdict(int)
        true = 0.0
        for nw, target in self.getExamples(n):

            pred = nw.predict(theta, roundoff=True)
            sse += nw.evaluate(theta, target, activate=False, roundoff=False)
            length = int((
                         nw.length + 1) / 2)  # number of leaves = the number of digits + the number of operators, which is #digits-1
            sspe[length] += nw.error(theta, target, activate=False, roundoff=True)
            lens[length] += 1
            if target == pred: true += 1
            if verbose == 2:
                print 'length:', length, (
                    'right' if target == pred else 'wrong'), 'prediction:', pred, 'target:', target, 'error:', nw.error(
                    theta,
                    target,
                    activate=False,
                    roundoff=True), '(' + str(
                    nw.error(theta, target, activate=False, roundoff=False)) + ')'

        mse = sse / n
        accuracy = true / n
        mspe = sum(sspe.values()) / n
        if verbose == 1: print '\tEvaluation on ' + name + ' data (' + str(n) + ' examples):'
        if verbose == 1: print '\tLoss (MSE):', mse, 'Accuracy:', accuracy, 'MSPE:', mspe
        if verbose == 1: print '\tMSPE per length: ', [
            (length, (sspe[length] / lens[length] if lens[length] > 0 else 'undefined')) for length in sspe.keys()]
        return {'loss (mse)': mse, 'accuracy': accuracy, 'mspe': mspe}


def getTestTBs(seed, kind, comparison=False):
    for name, mtb in arithmetics.test_treebank(seed):
        if kind == 'comparison':
            yield name, CompareClassifyTB(mtb.pairedExamples, comparison=comparison)
        elif kind == 'prediction':
            yield name, ScalarPredictionTB(mtb.examples)
        elif kind == 'RNN':
            yield name, RNNTB(mtb.examples)


def getTBs(seed, kind, comparison=False):
    data = {}
    for part in 'train', 'heldout':
        if part == 'train':
            mtb = arithmetics.training_treebank(seed)
        elif part == 'heldout':
            mtb = arithmetics.heldout_treebank(seed,
                                               languages={'L9_left': 500, 'L9_right': 500, 'L1': 5, 'L2': 50,
                                                          'L3': 150, 'L4': 200, 'L5': 300, 'L6': 400, 'L7': 500,
                                                          'L8': 500, 'L9': 500})
                                               #languages={'L9_left': 15000, 'L9_right': 15000, 'L1': 50, 'L2': 500,
                                               #           'L3': 1500, 'L4': 3000, 'L5': 5000, 'L6': 10000, 'L7': 15000,
                                               #           'L8': 15000, 'L9': 15000})
        if kind == 'comparison':
            data[part] = CompareClassifyTB(mtb.pairedExamples, comparison=comparison)
        elif kind == 'prediction':
            data[part] = ScalarPredictionTB(mtb.examples)
        elif kind == 'RNN':
            data[part] = RNNTB(mtb.examples)
    return data
