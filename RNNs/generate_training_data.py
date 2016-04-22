"""
Functions to generate training data for the
arithmetic task.
"""
from arithmetics import mathTreebank
from arithmetics import mathExpression
import keras.preprocessing.sequence
import re
import numpy as np


def generate_training_data(languages, architecture):
    """
    Take a treebank object and return numpy
    arrays with training data.
    """
    # generate treebank with examples
    treebank = generate_treebank(languages, architecture)

    # create map from digits and operators to integers
    digits = list(treebank.digits)
    N_digits = len(digits)
    operators = list(treebank.operators)
    N_operators = len(operators)
    digits.sort()
    N = N_digits + N_operators + 2
    d_map = dict(zip(digits+operators+['(',')'], np.arange(1, N+1)))

    # create empty input and targets
    X, Y = [], []

    # loop over examples
    for expression, answer in treebank.examples:
        input_seq = [d_map[i] for i in str(expression).split()]
        answer = str(answer)
        X.append(input_seq)
        Y.append(answer)

    # pad sequences to have the same length
    X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='int32')

    return X_padded, np.array(Y), N_digits, N_operators


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

