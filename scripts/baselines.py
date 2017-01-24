import numpy as np
import sys
from processing_arithmetics.arithmetics import MathTreebank, MathExpression
from processing_arithmetics.arithmetics.treebanks import treebank
from collections import OrderedDict, defaultdict
import random
import matplotlib
import matplotlib.pyplot as plt
import pickle

name = sys.argv[1]
f = open(name, 'w')

# write preamble
f.write('\\documentclass[a4paper]{article}\n')
f.write('\\usepackage{fullpage}')
f.write('\n\n\\begin{document}\n\n')

tabular_heading = '\\begin{tabular}[]{cc|p{6mm}p{6mm}p{6mm}p{8mm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}}\n'
row_names = 'input & stack & L1 & L2 & L3 & L4 & L5 & L6 & L7 & L8 & L9 & L9l & L9r\\\\\n'
tabular_ending = '\\end{tabular}'


digits = np.arange(-10,11)
operators = ['+', '-']
n = 5000

m = MathTreebank()
accuracies = []

noise_dict = {}

examples = m.generate_examples(operators=operators, digits=digits, n=10, lengths=[2])
examples_str = [example[0].to_string('infix') for example in examples]
true_outcomes = np.array([example[1] for example in examples])
predicted_outcomes = np.array([MathExpression.solve_locally(example[0], format='infix', operator_noise=0.03, digit_noise=0.03, stack_noise=0.0) for example in examples])


print(examples_str)
print(true_outcomes)
print(predicted_outcomes)
print(np.mean(np.square(true_outcomes-predicted_outcomes)))

raw_input()

exit()



for format in ['infix' , 'prefix', 'postfix']:
    f.write('\n\n\\section{'+format+'}\n')
    noise_dict[format] = {}
    print format

    for strat in [MathExpression.solve_locally, MathExpression.solve_recursively]:

        strategy_str = {MathExpression.solve_locally: "Incremental strategy", MathExpression.solve_recursively: "Recursive strategy"}[strat]

        print '\t', strategy_str

        if format == 'postfix' and strat == MathExpression.solve_locally:
            continue

        noise_dict[format][strat] = defaultdict(dict)

        f.write('\n\\subsection{' + strategy_str +'}\n\n')
        f.write(tabular_heading)
        f.write(row_names)

        for input_noise, stack_noise in [(i, j) for i in np.arange(0, 0.15, 0.03) for j in np.arange(0, 0.15, 0.03)]:
            print '\t\t', input_noise, stack_noise
            accuracies = [input_noise, stack_noise]
            # loop over lengths
            for length in np.arange(1, 10):
                print '\t\t\t', length
                examples = m.generate_examples(operators=operators, digits=digits, n=n, lengths=[length])
                true_outcomes = np.array([example[1] for example in examples])
                predicted_outcomes = np.array([strat(example[0], format=format, operator_noise=input_noise, digit_noise=input_noise, stack_noise=stack_noise) for example in examples])
                accuracies.append(np.mean(np.square(true_outcomes-predicted_outcomes)))
        
            # compute for L9r and L9l
            for branching in ['left', 'right']:
                examples = m.generate_examples(operators=operators, digits=digits, n=n, lengths=[9], branching=branching)
                true_outcomes = np.array([example[1] for example in examples])
                predicted_outcomes = np.array([strat(example[0], format=format, operator_noise=input_noise, digit_noise=input_noise, stack_noise=stack_noise) for example in examples])
                accuracies.append(np.mean(np.square(true_outcomes-predicted_outcomes)))
            noise_dict[format][strat][input_noise][stack_noise] = accuracies


            f.write('\n\n' + ' & '.join(['%.2f' % i for i in accuracies]) + '\\\\')

        f.write(tabular_ending)

f.write('\n\n\\end{document}\n\n')
f.close()

pickle.dump(noise_dict, open('noisy_strategies.dict', 'w'))

