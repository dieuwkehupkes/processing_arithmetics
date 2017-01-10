from __future__ import print_function
import numpy as np
import re
from collections import OrderedDict

from .MathTreebank import MathTreebank

languages_train = {'L1':3000, 'L2': 3000, 'L4':3000, 'L5':3000, 'L7':3000}
languages_heldout = {'L3':500, 'L6':800, 'L8':800}
languages_test = OrderedDict([('L1', 50), ('L2', 500), ('L3', 1500), ('L4', 3000), ('L5', 5000), ('L6', 10000), ('L7', 15000), ('L8', 15000), ('L9', 15000), ('L9_left', 15000), ('L9_right', 15000)])
train_languages_small = {'L1':3, 'L2': 3, 'L4':3, 'L5':3, 'L7':3}
test_languages_small = OrderedDict([('L9_left', 15), ('L9_right', 15), ('L1', 5), ('L2', 5), ('L3', 15), ('L4', 3), ('L5', 5), ('L6', 1), ('L7', 15), ('L8', 15), ('L9', 15)])
ds = np.arange(-10,11)
ops = ['+', '-']

def training_treebank(seed, languages=languages_train, digits=ds):
    np.random.seed(seed)
    m = MathTreebank(languages, digits=digits)
    return m


def heldout_treebank(seed, languages=languages_heldout, digits=ds):
    np.random.seed(seed)
    m = MathTreebank(languages, digits=digits)
    return m

def test_treebank(seed, languages=languages_test, digits=ds):
    np.random.seed(seed)
    for name, N in languages.items():
        yield name, MathTreebank(languages={name: N}, digits=digits)

def small_training_treebank(languages=train_languages_small, digits=ds):
    m = MathTreebank(languages, digits=digits)
    return m

def small_test_treebank(languages=test_languages_small, digits=ds):
    for name, N in languages.items():
        yield name, MathTreebank(languages={name: N}, digits=digits)

