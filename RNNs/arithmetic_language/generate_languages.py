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

# write one-hot vectors to file
pickle.dump(number2vec, open('one-hot_-19-19.pickle', 'wb'))

# generate all expressions with 2 numeric leaves (string representation)
L2 = []
L2_input_expressions = []
for expression in it.product(input_numbers, ['+', '-'], input_numbers):
    # string representation input part expression, compute outcome
    expression_string = '('+''.join(list(expression))+')'
    outcome = str(eval(expression_string))

    # store list representation for recursion
    input_expression = ['('] + list(expression) + [')']
    L2_input_expressions.append(input_expression)

    # generate full expression
    expr = list(expression) + ['=', outcome]
    L2.append(expr)

pickle.dump(L2, open('L2.pickle', 'wb'))

# generate all expressions with 3 numeric leaves whose expression is no larger than 60
L3_in, L3 = [], []
L3_left_branching = it.product(L2_input_expressions, ['+','-'], input_numbers)
L3_right_branching = it.product(input_numbers, ['+','-'], L2_input_expressions)
for expression in L3_left_branching:
    expression = expression[0] + list(expression[1:])
    L3_in.append(expression)

for expression in L3_right_branching:
    expression = list(expression[:-1]) + expression[-1]
    L3_in.append(expression)

for expression in L3_in:
    expression_string = ''.join(list(expression))
    outcome = eval(expression_string)
    expr = list(expression) + ['=', str(outcome)]
    L3.append(expr)

pickle.dump(L3, open('L3.pickle', 'wb'))

