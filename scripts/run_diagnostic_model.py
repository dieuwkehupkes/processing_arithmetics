from keras.models import load_model
import argparse
import pickle
import re
import numpy as np
from processing_arithmetics.sequential.architectures import DiagnosticClassifier
from processing_arithmetics.arithmetics.treebanks import treebank
from argument_transformation import max_length

"""
Get test results for a diagnostic model
"""

###################################################
# Set some params
digits = np.arange(-10, 11)
operators = ['+', '-']

###################################################
# Create argument parser

parser = argparse.ArgumentParser()
parser.add_argument("-models", type=str, nargs="*", help="Models to test")
parser.add_argument("-classifiers", required=True, nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical', 'intermediate_directly', 'depth', 'minus1depth', 'minus2depth', 'minus3depth', 'minus4depth', 'minus1depth_count'])

parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))
parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2)
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")
parser.add_argument("--output_name", help="Set output name")

args = parser.parse_args()

####################################################
# Set some params
languages_test              = [(name, tb) for name, tb in treebank(seed=args.seed_test, kind='test')]

results_all = {}

training_data = None
validation_data = None
test_data = None

for model in args.models:

    m = DiagnosticClassifier(model=model, classifiers=args.classifiers, copy_weights=['recurrent', 'embeddings', 'classifier'])

    # find format (for now assume it is in the title) and assure it is right
    format = re.search('postfix|prefix|infix', model).group(0)
    assert format == args.format

    # find recurrent layer
    layer_type = re.search('SimpleRNN|GRU|LSTM', model).group(0)

    # generate_test_data
    test_data = test_data or m.generate_test_data(data=languages_test, digits=digits, format=args.format)

    evaluation = m.test(test_data)
    eval_str = m.evaluation_string(evaluation)

    results_all[model] = evaluation

    print model, eval_str


# dump all results
if args.output_name:
    pickle.dump(results_all, open(args.output_name,'wb'))

