from processing_arithmetics.seqbased.architectures import ScalarPrediction
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from argument_transformation import get_architecture
import argparse
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pylab as plt

"""
Describe what this script does.
"""

###################################################
# Create argument parser

parser = argparse.ArgumentParser()

parser.add_argument("--models", required=True, nargs="*", help="Models for which to plot accuracies")
parser.add_argument("--architecture", required=True, type=get_architecture, help="Architecture of the model", choices=[ScalarPrediction])
parser.add_argument("-N", type=int, help="Number of examples to test for each language", default=5000)
parser.add_argument("--classifiers", nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical'])
parser.add_argument("--format", choices=['infix', 'prefix', 'postfix'], default='infix', help="format of arithmetic expressions")

metrics = ['mean_squared_prediction_error', 'binary_accuracy', 'mean_squared_error']

args = parser.parse_args()

#########################################################
# Set languages for plots
languages = OrderedDict(zip(['L1','L2','L3','L4','L5','L6','L7','L8','L9'], 9*[args.N]))
# languages = OrderedDict(zip(['L1_left','L2_left','L3_left','L4_left','L5_left','L6_left','L7_left','L8_left','L9_left'], 9*[args.N]))
digits = np.arange(-10,11)

# generate test data
treebanks = []
languages = [(name, MathTreebank({name: N} ,digits)) for name, N in languages.items()]

# compute mspe and binary accuracy on test data
strategy_results = {'mean_squared_error':[], 'binary_accuracy':[], 'mean_squared_prediction_error':[]}
for name, treebank in languages:
    correct = []
    se = []
    i = 0
    for expression, answer in treebank.examples:
        outcome = expression.solve_almost('infix')
        se.append(np.square(outcome - answer))
        correct.append(outcome == answer)

    # add outcome to dictionary
    strategy_results['mean_squared_prediction_error'].append(np.mean(se))
    strategy_results['mean_squared_error'].append(np.mean(se))
    strategy_results['binary_accuracy'].append(np.mean(correct))

# compute mspe and binary accuracy for models
results_all = dict()
for model in args.models:
    A = args.architecture(digits=digits)
    A.add_pretrained_model(model)
    test_data = A.generate_test_data(data=languages, digits=digits, format=args.format)

    results = A.test(test_data=test_data, metrics=metrics)
    # [(m1, val), (m2, val)]
    results_all[model] = results

# plot symbolic approaches
x = np.arange(len(languages))

for metric in metrics:
    if metric[-4:] == 'loss':
        continue

    # plot symbolic approaches
    if metric == 'mean_squared_error':
        y_max = 9*[0]
    if metric == 'mean_squared_prediction_error':
        y_max = 9*[0]
    elif metric == 'binary_accuracy':
        y_max = 9*[1]
    plt.plot(x, strategy_results[metric], label='Solve almost')
    plt.plot(x, y_max, label='Solve recursively')
    plt.plot(x, y_max, label='Solve incrementally')

    # plot different models
    for model in results_all:
        results_metric = [results_all[model][lang][metric] for lang in results_all[model]]
        plt.plot(x, results_metric, label=model)

    plt.title(metric)
    plt.xlabel("number of numeric elements")
    plt.ylabel(metric)
    plt.legend()

    plt.show()


