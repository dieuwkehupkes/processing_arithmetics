from __future__ import division
import numpy as np
import sys

'''
 Theta is an object that can hold the parameters of a complex neural network.
 A similar object 'Gradient' can be used to hold gradient values.
 The word embeddings are stored in a special object; WordMatrix
 All objects allow for pickling
'''

class Theta(dict):
    def __init__(self, dims, embeddings=None, vocabulary=['UNKNOWN'], seed=0):
        if dims is None:
            print 'No dimensions for initialization of theta'
            sys.exit()
        np.random.seed(seed)
        self.dims = dims
        self.compositionMatrices()

        # vocabulary = None, default = ('UNKNOWN', 0), dicItems = {}):
        # Install word embeddings as a WordMatrix object
        if embeddings is None:
            default = ('UNKNOWN', np.random.random_sample(self.dims['word']) * .2 - .1)
            self[('word',)] = WordMatrix(vocabulary, default = default, dicItems=[(word, np.random.random_sample(self.dims['word']) * .2 - .1) for word in vocabulary])
        else:
            self[('word',)] = embeddings


    def removeAll(self, toRemove=[]):
        # Remove all matrices and biases that belong to the categories in toRemove
        for key in self.keys():
            if key[0] in toRemove: del self[key]


    def extend4Classify(self, nChildren, nClasses, dComparison=0):
        # Add (or replace) all matices involved in classification
        self.removeAll(['classify', 'comparison'])
        if dComparison == 0:
            self.newMatrix(('classify', 'M'), None, (nClasses, nChildren * self.dims['inside']))
            self.newMatrix(('classify', 'B'), None, (nClasses,))
        else:
            if dComparison == -1:
                try:
                    dComparison = self.dims['comparison']
                except:
                    dComparison = (nChildren + 1) * self.dims['inside']
            self.newMatrix(('comparison', 'M'), None, (dComparison, nChildren * self.dims['inside']))
            self.newMatrix(('classify', 'M'), None, (nClasses, dComparison))
            self.newMatrix(('comparison', 'B'), None, (dComparison,))
            self.newMatrix(('classify', 'B'), None, (nClasses,))

    def extend4Prediction(self, dHidden=0):
        # Add (or replace) all matices involved in prediction
        self.removeAll(['predict', 'predictH'])
        if dHidden < 0:
            self.newMatrix(('predict', 'M'), None, (1, self.dims['inside']))
            self.newMatrix(('predict', 'B'), None, (1,))
        else:
            if dHidden == 0:
                try:
                    dHidden = self.dims['hidden']
                except:
                    dHidden = self.dims['inside']
            self.newMatrix(('predict', 'M'), None, (1, dHidden))
            self.newMatrix(('predict', 'B'), None, (1,))
            self.newMatrix(('predictH', 'M'), None, (dHidden, self.dims['inside']))
            self.newMatrix(('predictH', 'B'), None, (dHidden,))

    def compositionMatrices(self):
        din = self.dims['inside'] # local dimensionality variable
        try:
            minArity = self.dims['minArity']
        except:
            minArity = 1
        for arity in xrange(minArity, self.dims['maxArity'] + 1):
            lhs = '#X#'
            rhs = '(' + ', '.join(['#X#'] * arity) + ')'
            cat = 'composition'
            self.newMatrix((cat, lhs, rhs, 'I', 'M'), None, (din, arity * din))
            self.newMatrix((cat, lhs, rhs, 'I', 'B'), None, (din,))

    def reset(self, cats):
        for cat in cats:
            if isinstance(self[cat], WordMatrix):
                for word in self[cat].keys():
                    size = self[cat][word].shape
                    self[cat][word] = np.random.random_sample(size) * .2 - .1
            else:
                size = self[cat].shape
                self[cat] = np.random.random_sample(size) * .2 - .1

    def newMatrix(self, name, M=None, size=(0, 0), replace = False):
        if not replace and name in self:
            return

        if M is not None:
            self[name] = np.copy(M)
        else:
            self[name] = np.random.random_sample(size) * .2 - .1

    def norm(self):
        names = [name for name in self.keys() if name[-1] == 'M']
        return sum([np.linalg.norm(self[name]) for name in names]) / len(names)

    def gradient(self):
        molds = {}
        for key in self.keys():
            if isinstance(self[key], np.ndarray):
                molds[key] = np.shape(self[key])
        # initialize wordmatrix with zeroes default
        voc = self[('word',)].keys()
        defaultkey = 'UNKNOWN'
        defaultvalue = np.zeros_like(self[('word',)][defaultkey])
        wordM = WordMatrix(vocabulary=voc, default=(defaultkey, defaultvalue))
        return Gradient(molds, wordM)

    def __missing__(self, key):
        for fakeKey in generalizeKey(key): # find a more general version of key that is in theta (used with grammar rule specialized parameter)
            if fakeKey in self.keys():
                return self[fakeKey]
                break
        else:
            raise KeyError(str(key) + ' not in theta (missing).')

    def __iadd__(self, other):
        scalar = isinstance(other, int)
        for key in self:
            if isinstance(self[key], np.ndarray):
                if scalar:
                    self[key] = self[key] + other
                else:
                    self[key] = self[key] + other[key]
            elif isinstance(self[key], dict):
                for word in self[key]:
                    if scalar:
                        self[key][word] = self[key][word] + other
                    else:
                        self[key][word] = self[key][word] + other[key][word]
            else:
                print 'Inplace addition of theta failed:', key, 'of type', str(type(self[key]))
                sys.exit()
        return self

    def __add__(self, other):
        scalar = isinstance(other, int)

        newT = self.gradient()
        for key in self:
            if isinstance(self[key], np.ndarray):
                if scalar:
                    newT[key] = self[key] + other
                else:
                    newT[key] = self[key] + other[key]
            elif isinstance(self[key], dict):
                for word in self[key]:
                    if scalar:
                        newT[key][word] = self[key][word] + other
                    else: newT[key][word] = self[key][word] + other[key][word]
            else:
                print 'Inplace addition of theta failed:', key, 'of type', str(type(self[key]))
                sys.exit()
        return newT

    def __itruediv__(self, other):
        if isinstance(other, dict):
            th = True
        elif isinstance(other, int):
            th = False
        else:
            print 'unknown type of other in theta.itruediv'

        for key in self:
            if isinstance(self[key], np.ndarray):
                if th:
                    self[key] /= other[key]
                else:
                    self[key] /= other
            elif isinstance(self[key], dict):
                if th:
                    for word in other[key]:
                        self[key][word] /= other[key][word]
                else:
                    for word in self[key]:
                        self[key][word] /= other
            else:
                print 'Inplace division of theta failed:', key, 'of type', str(type(self[key]))
                sys.exit()
        return self

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def printDims(self):
        print 'Model dimensionality:'
        for key, value in self.dims.iteritems():
            print '\t' + key + ' - ' + str(value)

    def __str__(self):
        txt = '<<THETA>>'
        txt += ' words: ' + str(len(self[('word',)]))  # +str(self[('word',)].keys()[:5])
        return txt


class Gradient(Theta):
    def __init__(self, molds, wordM, kvs={}):
        self.molds = molds
        dict.__setitem__(self, ('word',), wordM)
        self.update(kvs)

    def __reduce__(self):
        return (self.__class__, (self.molds, self[('word',)], self.items()))

    def __missing__(self, key):
        if key in self.molds:
            self.newMatrix(key, np.zeros(self.molds[key]))
            return self[key]
        else:
            for fakeKey in generalizeKey(key):
                if fakeKey in self.molds:
                    self.newMatrix(fakeKey, np.zeros(self.molds[fakeKey]))
                    return self[fakeKey]
            else:
                print key, 'not in gradient(missing), and not able to create it.'
                return None

    def __setitem__(self, key, val):
        if key in self.molds:
            dict.__setitem__(self, key, val)
        else:
            for fakeKey in generalizeKey(key):
                if fakeKey in self.molds: dict.__setitem__(self, fakeKey, val)
                break
            else:
                raise KeyError(str(key) + 'not in gradient(setting), and not able to create it.')


class WordMatrix(dict):
    def __init__(self, vocabulary=None, default=('UNKNOWN', 0), dicItems=[]):
        self.voc = vocabulary
        dkey, dval = default
        if dkey not in self.voc: raise AttributeError("'default' must be in the vocabulary")
        self.default = dkey
        dict.__setitem__(self, self.default, dval)
        [dicItems.remove((k, v)) for (k, v) in dicItems if k == dkey]
        self.update(dicItems)

    def extendVocabulary(self, wordlist):
        for word in wordlist:
            self.voc.append(word)
            self[word] = self[self.default]

    def __setitem__(self, key, val):
        if self.default not in self: raise KeyError("Default not yet in the vocabulary: " + self.default)
        if key in self.voc:
            dict.__setitem__(self, key, val)
        else:
            dict.__setitem__(self, self.default, val)

    def erase(self):
        for key in self.keys():
            if key == self.default:
                continue
            else:
                del self[key]

    def __missing__(self, key):
        if key == self.default: raise KeyError("Default not yet in the vocabulary: " + self.default)
        if key in self.voc:
            self[key] = np.zeros_like(self[self.default])
            return self[key]
        else:
            return self[self.default]

    def __reduce__(self):
        return (self.__class__, (self.voc, (self.default, self[self.default]), self.items()))

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, got %d" % len(args))
            other = dict(args[0])
            for key, val in other.items():
                if key not in self.voc:
                    self[self.default] += val
                else:
                    self[key] += val
        for key, value in kwargs.items():
            if key not in self.voc:
                self[self.default] += val
            else:
                self[key] += val


def generalizeKey(key):
    if key[0] == 'composition' or key[0] == 'reconstruction':
        lhs = key[1]
        rhs = key[2]
        generalizedHead = '#X#'
        generalizedTail = '(' + ', '.join(['#X#'] * len(rhs[1:-1].split(', '))) + ')'
        return [key[:2] + (generalizedTail,) + key[3:], key[:1] + (generalizedHead, generalizedTail,) + key[3:]]
    else:
        return []
