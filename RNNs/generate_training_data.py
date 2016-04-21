"""
Functions to generate training data for the
arithmetic task.
"""
from arithmetics import mathTreebank
from arithmetics import mathExpression
import re

def generate_treebank(language, architecture=1):
    """
    Generate training examples.
    :param language_dict:   dictionary mapping languages to number of samples
    """
    treebank = mathTreebank()
    for name, N in languages.items():
        lengths, operators, branching = parse_language(name)
        treebank.addExamples(operators, branching=branching, lengths=lengths, n=N) 

    training_data = generate_training_data(treebank)
    return training_data

def generate_training_data(treebank):
    """
    Take a treebank object and return numpy
    arrays with training data.
    """
    treebank.write_to_file('treebank')
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

