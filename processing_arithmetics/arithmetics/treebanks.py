from __future__ import print_function
import numpy as np
import re
from collections import OrderedDict

from .MathTreebank import MathTreebank

languages = {
        'train':{L:3000 for L in ['L1','L2','L4','L5','L7']},
        'train_small':{L:3 for L in ['L1','L2','L4','L5','L7']},
        'heldout':{'L3':500, 'L6':800, 'L8':800},
        'heldout_small':{'L3':5, 'L6':8, 'L8':8},
        'test': OrderedDict([
            ('L1', 50), ('L2', 500), ('L3', 1500), ('L4', 3000), ('L5', 5000), ('L6', 10000), ('L7', 15000), ('L8', 15000), ('L9', 15000), ('L9_left', 15000), ('L9_right', 15000)
        ]),
        'test_small': OrderedDict([
            ('L9_left', 15), ('L9_right', 15), ('L1', 5), ('L2', 5), ('L3', 15), ('L4', 3), ('L5', 5), ('L6', 1), ('L7', 15), ('L8', 15), ('L9', 15)
        ])
}
ds = np.arange(-10,11)
ops = ['+', '-']

def treebank(seed, kind, digits=ds,debug=False):
    np.random.seed(seed)
    if kind == 'test':
        return test_treebank(digits, debug)
    else:
        return MathTreebank(languages[kind+('_small'if debug else '')], digits=digits)

def test_treebank(digits = ds, debug = False):
    for name, N in languages['test' + ('_small' if debug else '')].items():
        yield name, MathTreebank(languages={name: N}, digits=digits)
