# two strategies to solve the arithmetic expression symbolically

import numpy as np
from arithmetics import mathTreebank
import operator

def solveRecursiveExplicit(expr):
    outcome = 0
    op = operator.add
    numberstack = []
    operatorstack=[]
    symbols = iterate(expr)
    for symbol in symbols:
        if symbol == '(': # enter recursion
            # store what you were doing
            numberstack.append(outcome)
            operatorstack.append(op)
            # and start anew
            outcome = 0
            op = operator.add
        elif symbol == ')': # exit recursion
            # accumulate intermediate result with previous depth
            op = operatorstack.pop()
            outcome = op(numberstack.pop(),outcome)
        elif symbol == '+':
            op = operator.add
        elif symbol == '-':
            op = operator.sub
        else:
            # symbol is digit
            outcome = op(outcome,int(symbol))
    assert len(operatorstack)==0, len(numberstack)==0
    return outcome


def solveRecursive(expr):
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


def solveLocally(expr):
    """
    Input a syntactically correct bracketet
    expression, solve by counting brackets
    and depth.
    ( ( a + b ) )
    """
    result = 0
    subtracting = False

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
        outcome = solveRecursive(str(expression))
        print "outcomeRec = ", outcome
        outcome = solveLocally(str(expression))
        print "outcomeSeq = ", outcome
        outcome = solveRecursiveExplicit(str(expression))
        print "outcomeRecExplicit = ", outcome

        raw_input()


    
