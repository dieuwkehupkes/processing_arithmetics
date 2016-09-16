import sys
sys.path.insert(0, '../commonFiles') 
from test_model import test_model
from architectures import A1, A4
from collections import OrderedDict
import pickle
import numpy as np
from arithmetics import mathTreebank
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

prefs = 'models/GRU_A1_1_probe-1', 'models/GRU_A1_1_probe-2', 'models/GRU_A1_1_probe-3',
        'models/GRU_A1_2_probe-1', 'models/GRU_A1_2_probe-2', 'models/GRU_A1_2_probe-3',
        'models/SRN_A1_1_probe-1', 'models/SRN_A1_1_probe-2', 'models/SRN_A1_1_probe-3',
        'models/GRU_A4_A1_2_probe-1', 'models/GRU_A4_A1_2_probe-2', 'models/GRU_A4_A1_2_probe-3',
architecture = Probing
N = 5000
languages = OrderedDict(zip(['L1','L2','L3','L4','L5','L6','L7','L8','L9'], 9*[N]))
# metrics = ['mean_squared_prediction_error', 'binary_accuracy']
metrics = {'grammatical': ['binary_accuracy', 'binary_accuracy_ignore0'],
           'intermediate_locally': ['mean_squared_prediction_error', 'mean_squared_error'],
           'subtracting': ['binary_accuracy'],
           'intermediate_recursively': ['mean_squared_prediction_error', 'mean_squared_error'],
           'top_stack': ['mean_squared_error_ignore', 'mean_squared_prediction_error_ignore']}  
digits = np.arange(-10,11)
maxlen = 57

# generate test data
treebanks = []
for name, N in languages.items():
    treebank = mathTreebank({name: N}, digits) 
    treebanks.append((name, treebank))

# compute mspe and binary accuracy on test data
# strategy_results = {'mean_squared_prediction_error':[], 'binary_accuracy':[]}
# for name, treebank in treebanks:
#     correct = []
#     se = []
#     i = 0
#     for expression, answer in treebank.examples:
#         outcome = expression.solveAlmost()
#         se.append(np.square(outcome - answer))
#         correct.append(outcome == answer)
# 
#     # add outcome to dictionary
#     strategy_results['mean_squared_prediction_error'].append(np.mean(se))
#     strategy_results['binary_accuracy'].append(np.mean(correct))

# compute mspe and binary accuracy for models
for metric in metrics:
    results_all = dict()
    for pref in prefs:
        model_architecture = pref+'.json'
        model_weights = pref+'_weights.h5'
        model_dmap = 'models/dmap' 

        results = test_model(architecture=architecture, model_architecture=model_architecture, model_weights=model_weights, dmap=model_dmap, optimizer='adam', metrics=metrics, loss='mse', digits=digits, test_sets=treebanks)

        results_all[pref] = results

    # plot symbolic approaches
    x = np.arange(1,10)
    # if metric == 'mean_squared_prediction_error':
    #     y_max = 9*[0]
    # elif metric == 'binary_accuracy':
    #     y_max = 9*[1]

    # plt.plot(x, strategy_results[metric], label='SolveAlmost')
    # plt.plot(x, y_max, label='SolveRecursive')
    # plt.plot(x, y_max, label='SolveIncremental')

    # plot different models
    for model in results_all:
        results = results_all[model][metric].values()
        plt.plot(x, results, label=model)

    plt.title(metric)
    plt.xlabel("number of numeric elements")
    plt.ylabel(metric)
    plt.legend()

    plt.show()

