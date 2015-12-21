import pickle
import numpy as np
import itertools as it
from collections import OrderedDict

# generate sentences with different numbers of numeric leaves
# order of one-hot vectors: min_number, ....., max_number, +, -, (, ), =


min_input_number = -19
max_input_number = 19
min_number = -60
max_number = 60

# all numbers the network has representations of
numbers = [str(i) for i in xrange(min_number, max_number+1)]

# the numbers occuring in the input strings
input_numbers = [str(i) for i in xrange(min_input_number, max_input_number+1)]

# list of numbers and operators
numbers_operators = numbers + ['+', '-', '(', ')', '=']

# create dictionary with number vectors
identity = np.identity(len(numbers_operators))              
number2vec = OrderedDict([(numbers_operators[i], identity[i]) for i in xrange(len(identity))])

# generate all expressions with 2 numeric leaves and store in a dictionary with readable
# string representation
L2 = OrderedDict()
number2vecL2 = OrderedDict()
for expression in it.product(input_numbers, ['+', '-'], input_numbers):
    # string representation input part expression, compute outcome
    expression_string = '('+''.join(list(expression))+')'
    outcome = str(eval(expression_string))

    # generate vector representation expression_string and store in dict
    expression_matrix = [number2vec[item] for item in ['('] + list(expression) + [')']]
    number2vecL2[expression_string] = expression_matrix

    # generate full expression
    expr = list(expression) + ['=', outcome]
    expr_matrix = np.array([number2vec[item] for item in expr])
    L2[''.join(expr)] = expr_matrix

pickle.dump(L2, open('L2.pickle', 'wb'))

number2vec.update(number2vecL2)

# generate all expressions with 3 numeric leaves whose expression is no larger than 60
L3 = OrderedDict()
number2vecL3 = OrderedDict()
L3_left_branching = it.product(number2vecL2.keys(), ['+','-'], input_numbers)
L3_right_branching = it.product(input_numbers, ['+','-'], number2vecL2.keys())
for expression in it.chain(L3_left_branching, L3_right_branching):
    # string representation input expression, compute outcome
    expression_string = ''.join(list(expression))
    outcome = eval(expression_string)

    expr = list(expression) + ['=', str(outcome)]
    if outcome <= max_number and outcome >= min_number:
        expr_matrix = np.array([number2vec[item] for item in expr])
        L3[''.join(expr)] = expr_matrix
    else:
        pass

pickle.dump(L3, open('L3.pickle', 'wb'))

