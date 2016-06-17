"""
Functions to generate training data for the
arithmetic task.
"""
from arithmetics import mathTreebank

from auxiliary_functions import max_length
import keras.preprocessing.sequence
import re
import numpy as np
from collections import OrderedDict
import random
import pickle

def generate_dmap(digits, *languages):
    """
    Generate a map mapping the words in training and testset to one-hot vectors
    :param digits:      array with digits to be used
    :param languages: variable number of dictionaries mapping language names
    to numbers
    :return: dictionary mapping numbers and operators (str) to vectors
    """
    operators = set([])
    for language_dict in languages:
        if not language_dict:
            continue
        for name in language_dict.keys():
            plusmin = re.compile('\+')
            op = plusmin.search(name)
            if op:
                new_operators = [op.group()]
            else:
                new_operators = ['+','-']
            operators = operators.union(set(new_operators))

    # create map from digits and operators to integers
    operators = list(operators)
    N_operators = len(operators)
    digits = list(digits)
    digits.sort()
    digits = [str(i) for i in digits]
    N_digits = len(digits)
    N = N_digits + N_operators + 2
    # start counting at 1 to not ignore first word during training
    dmap = OrderedDict(zip(digits+operators+['(', ')'], np.arange(1, N+1)))
    return dmap, N_operators, N_digits

def string_to_vec(expressions, dmap, pad_to=None):
    """
    Generate a sequence of inputsequences for the network that
    matches the sequence of symbolic expressions inputted to the
    function
    :param expressions: list with string representations of arithmetic expressions
    :param dmap: dictionary describing how to map input symbols to integers
    :param pad_to: length of the input sequences
    :return:    np.array with inputsequences
    """
    X = np.array([[dmap[symbol] for symbol in expression.split(' ')] for expression in expressions])
    # pad sequences to have the same length
    assert pad_to == None or len(X[0]) <= pad_to, 'length test is %i, max length is %i. Test sequences should not be truncated' % (len(X[0]), pad_to)
    input_seqs = keras.preprocessing.sequence.pad_sequences(X, dtype='int32', maxlen=pad_to)
    return input_seqs


def generate_treebank(languages, digits):
    """
    Generate training examples.
    :param languages:   dictionary mapping languages to number of samples
    """
    treebank = mathTreebank()
    for name, N in languages.items():
        lengths, operators, branching = parse_language(name)
        treebank.add_examples(digits=digits, operators=operators, branching=branching, lengths=lengths, n=N)

    return treebank

def parse_language(language_str):
    """
    Give in a string for a language, return
    a tuple with arguments to generate examples.
    :return:    (#leaves, operators, branching)
    """
    # find # leaves
    nr = re.compile('[0-9]+')
    n = int(nr.search(language_str).group())

    # find operators
    plusmin = re.compile('\+')
    op = plusmin.search(language_str)
    if op:
        operators = [op.group()]
    else:
        operators = ['+','-']

    # find branchingness
    branch = re.compile('left|right')
    branching = branch.search(language_str)
    if branching:
        branching = branching.group()

    return [n], operators, branching

