from __future__ import division
import random
import NN as NN
from ..arithmetics import treebanks as arithmetics
from collections import defaultdict, Counter

class TreeBank():
    def __init__(self, examples):
        self.examples = examples

    def get_examples(self, n=0):
        if n == 0: n = len(self.examples)
        random.shuffle(self.examples)
        return self.examples[:n]


def confusion_s(matrix, labels):
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
        self.examples = self.convert_examples(examples, comparison)

    def convert_examples(self, items, comparison):
        examples = []

        for left, right, label in items:
            classifier = NN.Classifier(children=[NN.RNN(left).root, NN.RNN(right).root], labels=self.labels,
                                       comparison=comparison)
            examples.append((classifier, label))
        return examples

    def evaluate(self, theta, n=0, verbose=1):
        if n == 0: n = len(self.examples)
        error = 0.0
        true = 0.0
        diffs = []
        if verbose==1: confusion = defaultdict(Counter)
        for nw, target in self.get_examples(n):
            error += nw.evaluate(theta, target)
            prediction = nw.predict(theta=theta, activate = False)
            if verbose == 1: confusion[target][prediction] += 1
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
        if verbose == 1: print '\tLoss (cross entropy):', loss, ', accuracy:', accuracy, 'Confusion:'
        if verbose == 1: print confusionS(confusion, self.labels)
        if verbose > 0: print '\tAverage absolute difference of missclassified examples:', sum(diffs) / len(diffs)
        return {'loss (cross entropy)': loss, 'accuracy': accuracy}


# def getTestTBs(seed, kind, comparison=False, debug=False):
#     for name, mtb in arithmetics.test_treebank(seed,):
#         if kind == 'comparison':
#             yield name, CompareClassifyTB(mtb.pairedExamples(), comparison=comparison)
#         elif kind == 'RNN':
#             yield name, RNNTB(mtb.examples)

def data4comparison(seed, comparisonLayer=False, debug = False):
    data = {}
    for part in 'train', 'heldout':
        mtb = arithmetics.treebank(seed, kind=part, debug=debug)
        data[part] = CompareClassifyTB(mtb.paired_examples(), comparison=comparisonLayer)
    return data



# def getTBs(seed, kind='comparison', comparisonLayer=False, debug = False):
#     data = {}
#     for part in 'train', 'heldout':
#         mtb = arithmetics.treebank(seed,kind=part,debug=debug)
#         if kind == 'comparison':
#             data[part] = CompareClassifyTB(mtb.pairedExamples(), comparison=comparisonLayer)
#         elif kind == 'RNN':
#             data[part] = RNNTB(mtb.examples)
#     return data

def data4prediction(theta, seed, debug= False):
    all_data = defaultdict(lambda: defaultdict(list))



    for part in 'train', 'heldout':
        mtb = arithmetics.treebank(seed, kind=part, debug=debug)
        for nw, target in RNNTB(mtb.examples).examples: #getTBs(seed=seed, kind='RNN',debug=debug)['train'].get_examples():
            a = nw.activate(theta)
            all_data['X_' + part]['all'].append(a)
            all_data['Y_' + part]['all'].append(target)
            all_data['strings_' + part]['all'].append(str(nw))
    mtbt = arithmetics.treebank(seed, kind='test', debug=debug)
    for lan, tb in mtbt:
        for nw, target in RNNTB(mtb.examples).get_examples():
            a = nw.activate(theta)
            all_data['X_test'][lan].append(a)
            all_data['Y_test'][lan].append(target)
            all_data['strings_test'][lan].append(str(nw))
    return dict(all_data)
