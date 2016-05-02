"""
Functions to generate training data for the
arithmetic task.
"""
from arithmetics import mathTreebank
from arithmetics import mathExpression
import keras.preprocessing.sequence
import re
import numpy as np
import random


def generate_training_data(languages, architecture, pad_to=None):
    """
    Take a treebank object and return numpy
    arrays with training data.
    :param languages:       dictionary mapping languages (str name) to numbers
    :param architecture:    architecture for which to generate training data
    :param pad_to:          length to pad training data to
    :return:                tuple, input, output, number of digits, number of operators
                            map from input symbols to integers
    """
    # generate treebank with examples
    treebank = generate_treebank(languages, architecture)
    random.shuffle(treebank.examples)

    # create map from digits and operators to integers
    digits = list(treebank.digits)
    N_digits = len(digits)
    operators = list(treebank.operators)
    N_operators = len(operators)
    digits.sort()
    N = N_digits + N_operators + 2
    d_map = dict(zip(digits+operators+['(',')'], np.arange(0, N)))

    # create empty input and targets
    X, Y = [], []

    # loop over examples
    for expression, answer in treebank.examples:
        input_seq = [d_map[i] for i in str(expression).split()]
        answer = str(answer)
        X.append(input_seq)
        Y.append(answer)

    # pad sequences to have the same length
    assert pad_to == None or len(X[0]) <= pad_to, 'length test is %i, max length is %i. Test sequences should not be truncated' % (len(X[0]), pad_to)
    X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='int32', maxlen=pad_to)

    return X_padded, np.array(Y), N_digits, N_operators, d_map


def generate_treebank(languages, architecture):
    """
    Generate training examples.
    :param language_dict:   dictionary mapping languages to number of samples
    """
    treebank = mathTreebank()
    for name, N in languages.items():
        lengths, operators, branching = parse_language(name)
        treebank.addExamples(operators, branching=branching, lengths=lengths, n=N) 

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

    return ([n], operators, branching)

if __name__ == '__main__':
    languages = {'L_3lb-':10, 'L_4+':5}
    generate_treebank(languages)

