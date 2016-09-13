# two strategies to solve the arithmetic expression symbolically

import numpy as np
from arithmetics import mathTreebank
import operator

def solveRecursiveExplicit(expr):
    outcome = 0
    op = operator.add
    numberstack = []
    operatorstack = []
    symbols = iterate(expr)
    for symbol in symbols:
        if symbol == '(':  # enter recursion
            # store what you were doing
            numberstack.append(outcome)
            operatorstack.append(op)
            # and start anew
            outcome = 0
            op = operator.add
        elif symbol == ')':  # exit recursion
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
    assert len(operatorstack) == 0, len(numberstack) == 0
    return outcome


def solveRecursive(expr, return_sequences=False):
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
            op = operator.add
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


def solveLocally(expr, return_sequences=False):
    """
    Input a syntactically correct bracketet
    expression, solve by counting brackets
    and depth.
    """
    result = 0
    brackets = []
    subtracting = False

    symbols = iterate(expr)

    for symbol in symbols:
        
        if symbol[-1].isdigit():
            digit = int(symbol)
            if subtracting:
                result -= digit
            else:
                result += digit

        elif symbol == '(':
            brackets.append(subtracting)

        elif symbol == ')':
            brackets.pop(-1)
            try:
                subtracting = brackets[-1]
            except IndexError:
                # end of sequence
                pass

        elif symbol == '+':
            pass

        elif symbol == '-':
            subtracting = not subtracting

    return result

def iterate(expression):
    """
    Iterate over symbolic expression.
    """
    symbols = expression.split()
    # print(symbols)
    return symbols

def solveAlmost(expr):
    """
    Input a syntactically correct bracketet
    expression, solve by counting brackets
    and depth.
    ( ( a + b ) )
    """
  
    symbols = iterate(expr)
  
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


if __name__ == '__main__':
    from collections import OrderedDict
    import matplotlib.pylab as plt
    m = mathTreebank()
    # examples = m.generateExamples(operators=['+','-'], digits=np.arange(10), n=5000, lengths=[6,7,8,9])
    # x = '( 1 - ( ( ( 0 + 0 ) - 5 ) + ( 3 + 4 ) ) )'
    # print(x)
    # print(solveLocally(x))
    # print(eval(x))
    # exit()
    percentage_correct = OrderedDict([])

    for i in np.arange(1,21):
        examples = m.generateExamples(operators=['+','-'], digits=np.arange(-10, 10), n=5000, lengths=[i])
        correct = 0

        for expression, answer in examples:
            outcome = solveAlmost(str(expression))
            if outcome == answer:
                correct += 1

        percentage_correct[i] = correct/float(50)
    
    plt.plot(percentage_correct.keys(), percentage_correct.values())
    plt.title("Percentage correct for approximate strategy")
    plt.xlabel("Expression length")
    plt.ylabel("Percentage correct")
    plt.axis([0, 20, 0, 110])
    plt.show()

    exit()


    for expression, answer in examples:
        outcome = solveRecursive(str(expression))
        if outcome != answer:
            print('\n',  str(expression), '=', str(answer))
            print("computed outcome is:", outcome)
            raw_input()
        # print("outcomeRec = ", outcome)
        # outcome = solveLocally(str(expression))
        # print("outcomeSeq = ", outcome)
        # outcome = solveRecursiveExplicit(str(expression))
        # print("outcomeRecExplicit = ", outcome)

        # raw_input()


    
