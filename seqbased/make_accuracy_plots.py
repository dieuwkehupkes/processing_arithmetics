import sys
from test_model import test_model
from architectures import A1, A4
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../commonFiles') 
from arithmetics import mathTreebank

prefs = 'models/GRU_A1_1', 'models/GRU_A1_2'
architecture = A1
N = 5000
languages = OrderedDict(zip(['L1','L2','L3'],3*[N]))
metrics = ['mean_squared_prediction_error', 'binary_accuracy']
digits = np.arange(-10,11)
maxlen=57

# generate test data
treebanks = []
for name, N in languages.items():
    treebank = mathTreebank({name: N}, digits) 
    treebanks.append((name, treebank))

# compute mspe and binary accuracy on test data
strategy_results = {'mean_squared_prediction_error':[], 'binary_accuracy':[]}
for name, treebank in treebanks:
    correct = []
    se = []
    i = 0
    for expression, answer in treebank.examples:
        outcome = expression.solveAlmost()
        se.append(np.square(outcome - answer))
        correct.append(outcome==answer)

    # add outcome to dictionary
    strategy_results['mean_squared_prediction_error'].append(np.mean(se))
    strategy_results['binary_accuracy'].append(np.mean(correct))

# compute mspe and binary accuracy for models
for pref in prefs:
    model_architecture = pref+'.json'
    model_weights = pref+'_weights.h5'
    model_dmap = pref+'.dmap' 

    results_all = dict()

    results = test_model(architecture=architecture, model_architecture=model_architecture, model_weights=model_weights, dmap=model_dmap, optimizer='adam', metrics=metrics, loss='mse', digits=digits, test_sets=treebanks)

    results_all[pref] = results

    # generate plots
    for metric in metrics:
        x = np.arange(3)
        y_max = 3*[1]

        # plot symbolic approaches
        plt.plot(x, strategy_results[metric], label='SolveAlmost')
        plt.plot(x, y_max, label='SolveRecursive')
        plt.plot(x, y_max, label='SolveIncremental')

        # plot different models
        for model in  results_all:
            results = results_all[model][metric].values()
            plt.plot(x, results, label=model)

        plt.legend()

plt.show()











