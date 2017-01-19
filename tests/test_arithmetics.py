import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank

@pytest.fixture(params=[
    'infix', 'prefix'
])
def format(request):
    return request.param



def test_solve_locally(format):
    _test_solve_locally(format)


def test_solve_recursively(format):
    _test_solve_recursively(format)


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
            outcome = expression.solve_locally(format=format)
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
            outcome = expression.solve_recursively(format=format)
            if outcome != answer:
                incorrect += 1

    assert incorrect == 0

