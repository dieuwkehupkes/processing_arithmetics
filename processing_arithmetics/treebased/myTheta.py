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
        self.composition_matrices()

        # vocabulary = None, default = ('UNKNOWN', 0), dic_items = {}):
        # Install word embeddings as a WordMatrix object
        if embeddings is None:
            default = ('UNKNOWN', np.random.random_sample(self.dims['word']) * .2 - .1)
            self[('word',)] = WordMatrix(vocabulary, default = default, dic_items=[(word, np.random.random_sample(self.dims['word']) * .2 - .1) for word in vocabulary])
        else:
            self[('word',)] = embeddings


    def remove_all(self, to_remove=[]):
        # Remove all matrices and biases that belong to the categories in to_remove
        for key in self.keys():
            if key[0] in to_remove: del self[key]


    def extend4Classify(self, n_children, n_classes, d_comparison=0):
        # Add (or replace) all matices involved in classification
        self.remove_all(['classify', 'comparison'])
        if d_comparison == 0:
            self.new_matrix(('classify', 'M'), None, (n_classes, n_children * self.dims['inside']))
            self.new_matrix(('classify', 'B'), None, (n_classes,))
        else:
            if d_comparison == -1:
                try:
                    d_comparison = self.dims['comparison']
                except:
                    d_comparison = (n_children + 1) * self.dims['inside']
            self.new_matrix(('comparison', 'M'), None, (d_comparison, n_children * self.dims['inside']))
            self.new_matrix(('classify', 'M'), None, (n_classes, d_comparison))
            self.new_matrix(('comparison', 'B'), None, (d_comparison,))
            self.new_matrix(('classify', 'B'), None, (n_classes,))

    def extend4Prediction(self, d_hidden=0):
        # Add (or replace) all matices involved in prediction
        self.remove_all(['predict', 'predict_h'])
        if d_hidden < 0:
            self.new_matrix(('predict', 'M'), None, (1, self.dims['inside']))
            self.new_matrix(('predict', 'B'), None, (1,))
        else:
            if d_hidden == 0:
                try:
                    d_hidden = self.dims['hidden']
                except:
                    d_hidden = self.dims['inside']
            self.new_matrix(('predict', 'M'), None, (1, d_hidden))
            self.new_matrix(('predict', 'B'), None, (1,))
            self.new_matrix(('predict_h', 'M'), None, (d_hidden, self.dims['inside']))
            self.new_matrix(('predict_h', 'B'), None, (d_hidden,))

    def composition_matrices(self):
        din = self.dims['inside'] # local dimensionality variable
        try:
            min_arity = self.dims['min_arity']
        except:
            min_arity = 1
        for arity in xrange(min_arity, self.dims['max_arity'] + 1):
            lhs = '#X#'
            rhs = '(' + ', '.join(['#X#'] * arity) + ')'
            cat = 'composition'
            self.new_matrix((cat, lhs, rhs, 'I', 'M'), None, (din, arity * din))
            self.new_matrix((cat, lhs, rhs, 'I', 'B'), None, (din,))

    def reset(self, cats):
        for cat in cats:
            if isinstance(self[cat], WordMatrix):
                for word in self[cat].keys():
                    size = self[cat][word].shape
                    self[cat][word] = np.random.random_sample(size) * .2 - .1
            else:
                size = self[cat].shape
                self[cat] = np.random.random_sample(size) * .2 - .1

    def new_matrix(self, name, M=None, size=(0, 0), replace = False):
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
        word_m = WordMatrix(vocabulary=voc, default=(defaultkey, defaultvalue))
        return Gradient(molds, word_m)

    def __missing__(self, key):
        for fake_key in generalize_key(key): # find a more general version of key that is in theta (used with grammar rule specialized parameter)
            if fake_key in self.keys():
                return self[fake_key]
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

        new_t = self.gradient()
        for key in self:
            if isinstance(self[key], np.ndarray):
                if scalar:
                    new_t[key] = self[key] + other
                else:
                    new_t[key] = self[key] + other[key]
            elif isinstance(self[key], dict):
                for word in self[key]:
                    if scalar:
                        new_t[key][word] = self[key][word] + other
                    else: new_t[key][word] = self[key][word] + other[key][word]
            else:
                print 'Inplace addition of theta failed:', key, 'of type', str(type(self[key]))
                sys.exit()
        return new_t

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

    def print_dims(self):
        print 'Model dimensionality:'
        for key, value in self.dims.iteritems():
            print '\t' + key + ' - ' + str(value)

    def __str__(self):
        txt = '<<THETA>>'
        txt += ' words: ' + str(len(self[('word',)]))  # +str(self[('word',)].keys()[:5])
        return txt


class Gradient(Theta):
    def __init__(self, molds, word_m, kvs={}):
        self.molds = molds
        dict.__setitem__(self, ('word',), word_m)
        self.update(kvs)

    def __reduce__(self):
        return (self.__class__, (self.molds, self[('word',)], self.items()))

    def __missing__(self, key):
        if key in self.molds:
            self.new_matrix(key, np.zeros(self.molds[key]))
            return self[key]
        else:
            for fake_key in generalize_key(key):
                if fake_key in self.molds:
                    self.new_matrix(fake_key, np.zeros(self.molds[fake_key]))
                    return self[fake_key]
            else:
                print key, 'not in gradient(missing), and not able to create it.'
                return None

    def __setitem__(self, key, val):
        if key in self.molds:
            dict.__setitem__(self, key, val)
        else:
            for fake_key in generalize_key(key):
                if fake_key in self.molds: dict.__setitem__(self, fake_key, val)
                break
            else:
                raise KeyError(str(key) + 'not in gradient(setting), and not able to create it.')


class WordMatrix(dict):
    def __init__(self, vocabulary=None, default=('UNKNOWN', 0), dic_items=[]):
        self.voc = vocabulary
        dkey, dval = default
        if dkey not in self.voc: raise AttributeError("'default' must be in the vocabulary")
        self.default = dkey
        dict.__setitem__(self, self.default, dval)
        [dic_items.remove((k, v)) for (k, v) in dic_items if k == dkey]
        self.update(dic_items)

    def extend_vocabulary(self, wordlist):
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


def generalize_key(key):
    if key[0] == 'composition' or key[0] == 'reconstruction':
        lhs = key[1]
        rhs = key[2]
        generalized_head = '#X#'
        generalized_tail = '(' + ', '.join(['#X#'] * len(rhs[1:-1].split(', '))) + ')'
        return [key[:2] + (generalized_tail,) + key[3:], key[:1] + (generalized_head, generalized_tail,) + key[3:]]
    else:
        return []
