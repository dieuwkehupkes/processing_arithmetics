import pytest
import numpy as np
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from processing_arithmetics.arithmetics.MathExpression import MathExpression as M

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
    digits = np.arange(-10,11)
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

def _test_compute_minus1depth_1():
    expr = '( ( 1 - 7 ) + ( ( 5 - ( ( 8 - 7 ) + ( 3 + 4 ) ) )  + 7 ) )'
    depth = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) 
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0])

    e = M.fromstring(expr)

    assert e.get_minus_depths(1) == depth
    assert e.depth_counts[1] == counts

def _test_compute_minus1depth_2():

    expr = '( ( ( 3 + 3 ) - ( ( 2 + 3 ) + 0 ) ) + 0 )'
    depth = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0])
    e = M.fromstring(expr)

    assert e.get_minus_depths(1) == depth
    assert e.depth_counts[1] == counts

