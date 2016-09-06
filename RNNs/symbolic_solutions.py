# two strategies to solve the arithmetic expression symbolically

import numpy as np
from arithmetics import mathTreebank

def solveRecursive(exp):
    raise NotImplementedError()

def solveSequential(expr):
    """
    Input a syntactically correct bracketet
    expression, solve by counting brackets
    and depth.
    ( ( a + b ) )
    """

    symbols = iterate(expr)

    result = 0
    subtracting = False

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
            if subtracting:
                subtracting = False

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
        outcome = solveSequential(str(expression))
        print "outcome = ", outcome
        raw_input()
    
