from __future__ import print_function

from processing_arithmetics.sequential.analyser import visualise_hidden_layer
from processing_arithmetics.sequential.architectures import Training, ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.arithmetics.treebanks import treebank

from argument_transformation import get_architecture, get_hidden_layer, max_length

import argparse
import pickle
import re
import numpy as np


###################################################
# Create argument parser

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("-architecture", required=True, type=get_architecture, help="Type of architecture used during training: scalar prediction, comparison training, seq2seq or a diagnostic classifier", choices=[ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq])
parser.add_argument("-models", required=True, nargs='*', type=str, help="Filenames with extension h5 containing a keras model")
parser.add_argument("-classifiers", nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical', 'intermediate_directly', 'depth', 'minus1depth', 'minus2depth', 'minus3depth', 'minus4depth'])
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")

parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("-metrics", nargs='*', required=True, help="Add if you want to test metrics other than the default ones. In case of multiple outputs, all metrics will be applied to all outputs")

parser.add_argument("-save_to", help="Save to file name")

args = parser.parse_args()

languages_test = [(name, tb) for name, tb in treebank(seed=args.seed_test, kind='test')]
digits = np.arange(-10, 11)
operators = ['+', '-']
test_data = None

results_all = {}

for model in args.models:
    architecture = args.architecture(digits=digits, operators=operators, classifiers=args.classifiers)
    architecture.add_pretrained_model(model=model)
    test_data = test_data or architecture.generate_test_data(data=languages_test, digits=digits)
    results = architecture.test(test_data, metrics=args.metrics)
    results_all[model] = results
    print(architecture.evaluation_string(results))

if args.save_to:
    pickle.dump(results_all, open(args.save_to+'.results','wb'))
