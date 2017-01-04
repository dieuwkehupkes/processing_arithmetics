from __future__ import print_function

from processing_arithmetics.seqbased.analyser import visualise_hidden_layer
from processing_arithmetics.seqbased.architectures import Training, ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.arithmetics.treebanks import test_treebank

from argument_transformation import get_architecture, get_hidden_layer, max_length

import argparse
import pickle
import re
import numpy as np


###################################################
# Create argument parser

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("architecture", type=get_architecture, help="Type of architecture used during training: scalar prediction, comparison training, seq2seq or a diagnostic classifier", choices=[ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq])
parser.add_argument("model", type=str, help="Filename with extension h5 containing a keras model")
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")

parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("metrics", nargs='*', help="Add if you want to test metrics other than the default ones. In case of multiple outputs, all metrics will be applied to all outputs")

args = parser.parse_args()

languages_test = [(name, treebank) for name, treebank in test_treebank(seed=args.seed_test)]
digits = np.arange(-10, 11)
operators = ['+', '-']

architecture = args.architecture(digits=digits, operators=operators)
architecture.add_pretrained_model(model=args.model)
test_data = architecture.generate_test_data(data=languages_test, digits=digits)
architecture.test(test_data, metrics=args.metrics)  # TODO test if this also works for multilpe outputs
