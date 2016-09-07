# two strategies to solve the arithmetic expression symbolically

import numpy as np
from arithmetics import mathTreebank
import operator

def solveRecursive(expr, return_sequence=False):
    """
    Solve expression recursively.
    """
    stack = [[operator.add, 0]]
    op = operator.add

    symbols = iterate(expr)

    for symbol in symbols:
        if symbol == '(':
            # push new element on stack
            stack.append([op, 0])
        elif symbol == ')':
            # combine last stack item with
            # one but last stack item
            stack_op, outcome = stack.pop(-1)
            stack[-1][1] = stack_op(stack[-1][1], outcome)
        elif symbol == '+':
            op = operator.add
        elif symbol == '-':
            op = operator.sub
        else:
            # number is digit
            stack[-1][1] = op(stack[-1][1], int(symbol))

    assert len(stack) == 1, "expression not grammatical"

    return stack[0][1]
        

def solveLocally(expr, return_sequence=False):
    """
    Input a syntactically correct bracketet
    expression, solve by counting brackets
    and depth.
    ( ( a + b ) )
    """
    result = 0
    subtracting = 0

    symbols = iterate(expr)

    for symbol in symbols:
        if symbol[-1].isdigit():
            digit = int(symbol)
            if subtracting:
                result -= digit
            else:
                result += digit
        elif symbol == '-':
            subtracting = not subtracting

        if symbol == ')':
            if subtracting > 0:
                subtracting -= 1

    return result


def iterate(expression):
    """
    Iterate over symbolic expression.
    """
    symbols = expression.split()
    # print symbols
    return symbols


if __name__ == '__main__':
    m = mathTreebank()
    examples = m.generateExamples(operators=['+','-'], digits=np.arange(-10, 10), n=5, lengths=[2,3,4])
    for expression, answer in examples:
        print '\n',  str(expression), '=', str(answer)
        outcome = solveRecursive(str(expression))
        print "outcome = ", outcome
        raw_input()
    
