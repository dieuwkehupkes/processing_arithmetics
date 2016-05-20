"""
Functions to generate training data for the
arithmetic task.
"""
from arithmetics import mathTreebank
from auxiliary_functions import max_length
import keras.preprocessing.sequence
import re
import numpy as np
import random
import pickle

def generate_test_data(languages, architecture, dmap, digits, pad_to=None):
    """
    Take a dictionary that maps language names to number of sentences and return numpy array
    with test data.
    :param languages:       dictionary mapping language names to numbers
    :param architecture:    architecture for which to generate test data
    :param pad_to:          desired length of test sequences
    :return:                list of tuples containing test set sames, inputs and targets
    """
    test_data = []
    for name, N in languages.items():
        X, Y = [], []
        treebank = mathTreebank()
        lengths, operators, branching = parse_language(name)
        treebank.add_examples(operators, digits=digits, branching=branching, lengths=lengths, n=N)

        for expr, answ in treebank.examples:
            input_seq = [dmap[i] for i in str(expr).split()]
            answer = str(answ)
            X.append(input_seq)
            Y.append(answer)

        # pad sequences to have the same length
        assert pad_to == None or len(X[0]) <= pad_to, 'length test is %i, max length is %i. Test sequences should not be truncated' % (len(X[0]), pad_to)
        X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='int32', maxlen=pad_to)
        test_data.append((name, X_padded, np.array(Y)))

    return test_data


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
    dmap = dict(zip(digits+operators+['(', ')'], np.arange(1, N+1)))
    return dmap, N_operators, N_digits


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

if __name__ == '__main__':
    languages = {'L_3':500}
    digits = np.arange(-19, 20)
    dmap, N_operators, N_digits = generate_dmap(digits, languages)
    test_data = generate_training_data(languages, architecture='A1', dmap=dmap, digits=digits, pad_to=max_length(7))
    pickle.dump(test_data, open('test_sets/L3_500.test', 'wb'))
    # pickle.dump(dmap, open('model_test.dmap', 'wb'))

