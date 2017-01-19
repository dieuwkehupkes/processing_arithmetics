import numpy as np
import sys
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.arithmetics.treebanks import treebank
from collections import OrderedDict
import random
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    digits = np.arange(-10,11)
    operators = ['+', '-']
    operator_noise = 0.01
    digit_noise = 0.01
    stack_noise = 0.1
    format = 'infix'

    m = MathTreebank()
    accuracies = []

    for length in np.arange(3, 10):
        print length
        examples = m.generate_examples(operators=operators, digits=digits, n=5, lengths=[length])
        true_outcomes = np.array([example[1] for example in examples])
        predicted_outcomes = np.array([example[0].solve_recursively(format=format, operator_noise=operator_noise, digit_noise=digit_noise, stack_noise=stack_noise) for example in examples])
        accuracies.append(np.mean(np.square(true_outcomes-predicted_outcomes)))

    for branching in ['left', 'right']:
        examples = m.generate_examples(operators=operators, digits=digits, n=500, lengths=[9], branching=branching)
        true_outcomes = np.array([example[1] for example in examples])
        predicted_outcomes = np.array([example[0].solve_locally(format=format, operator_noise=operator_noise, digit_noise=digit_noise, stack_noise=stack_noise) for example in examples])
        accuracies.append(np.mean(np.square(true_outcomes-predicted_outcomes)))

print accuracies

