import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank

def test_solve_locally_infix():
    _test_solve_locally('infix')

def test_solve_locally_prefix():
    _test_solve_locally('prefix')

def test_solve_recursively_infix():
    _test_solve_recursively('infix')

def test_solve_recursively_prefix():
    _test_solve_recursively('prefix')

def test_solve_recursively_postfix():
    _test_solve_recursively('postfix')


def _test_solve_locally(format):
    digits = np.arange(-10,10)
    operators = ['+', '-']
    m = MathTreebank()
    incorrect = 0
    for length in np.arange(3,10):
        examples = m.generate_examples(operators=operators, digits=digits, n=500, lengths=[length])
        for expression, answer in examples:
            outcome = expression.solveLocally(format=format)
            if outcome != answer:
                incorrect += 1

    assert incorrect == 0

def _test_solve_recursively(format):
    digits = np.arange(-10,10)
    operators = ['+', '-']
    m = MathTreebank()
    incorrect = 0
    for length in np.arange(3,10):
        examples = m.generate_examples(operators=operators, digits=digits, n=500, lengths=[length])
        for expression, answer in examples:
            outcome = expression.solveRecursively(format=format)
            if outcome != answer:
                incorrect += 1

    assert incorrect == 0

