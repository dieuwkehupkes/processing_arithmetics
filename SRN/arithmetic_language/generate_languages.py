import pickle
import numpy as np
import itertools as it

# generate sentences with different numbers of numeric leaves
# order of one-hot vectors: min_number, ....., max_number, +, -, (, ), =


min_number = -2
max_number = 2
min_input_number = -1
max_input_number = 1

numbers = [str(i) for i in xrange(min_number, max_number+1)]
input_numbers = [str(i) for i in xrange(min_input_number, max_input_number+1)]
numbers_operators = numbers + ['+', '-', '(', ')', '=']
vectors = np.identity(len(numbers_operators))

# generate all expressions with 2 numeric leaves
L2 = []
for expression in it.product(input_numbers, ['+', '-'], input_numbers):
    outcome = str(eval(''.join(list(expression))))
    expr = list(expression) + ['=', outcome]
    expr_vector = np.array([vectors[numbers_operators.index(item)] for item in expr])
    L2.append(expr_vector)

pickle.dump(L2, open('L2.pickle', 'wb'))

# generate all expressions with 3 numeric leaves
# TODO implement generate expressions with 3 numeric leaves
# I think it makes sense to use the L2 phrases and combine them with another numeric leave
