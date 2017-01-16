from keras.models import load_model
import argparse
import pickle
import numpy as np
from processing_arithmetics.sequential.architectures import DiagnosticClassifier
from processing_arithmetics.arithmetics.treebanks import training_treebank, test_treebank, heldout_treebank
from argument_transformation import max_length

"""
Train a diagnostic classifier for an existing model.
"""

###################################################
# Set some params
digits = np.arange(-10, 11)
operators = ['+', '-']

###################################################
# Create argument parser

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Model to diagnose")
parser.add_argument("save_to", help="Save model to filename")
parser.add_argument("--classifiers", required=True, nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical'])
parser.add_argument("--nb_epochs", type=int, required=True)

parser.add_argument("--seed", type=int, help="Set random seed", default=8)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--dropout", help="Set dropout fraction", default=0.0)
parser.add_argument("-b", "--batch_size", help="Set batch size", default=24)
parser.add_argument("--val_split", help="Set validation split", default=0.1)

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))

args = parser.parse_args()

####################################################
# Set some params
languages_train             = training_treebank(seed=args.seed)
languages_val              = heldout_treebank(seed=args.seed)
languages_test              = [(name, treebank) for name, treebank in test_treebank(seed=args.seed_test)]

training = DiagnosticClassifier(digits=digits, operators=operators, model=args.model, classifiers=args.classifiers)

training_data = training.generate_training_data(languages_train, format=args.format)
validation_data = training.generate_training_data(languages_val, format=args.format)

training.train(training_data=training_data, validation_data=validation_data, 
        validation_split=args.val_split, batch_size=args.batch_size,
        epochs=args.nb_epochs, verbosity=1, filename=args.save_to,
        save_every=False)


######################################################################################
# Test model and write to file

# generate_test_data
test_data = training.generate_test_data(data=languages_test, digits=digits, format=args.format)

training.test(test_data)
