from keras.models import load_model
import argparse
import pickle
import re
import numpy as np
from processing_arithmetics.sequential.architectures import DiagnosticClassifier, DCgates
from processing_arithmetics.arithmetics.treebanks import treebank
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
parser.add_argument("-models", type=str, nargs="*", help="Models to diagnose")
parser.add_argument("-classifiers", required=True, nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical', 'intermediate_directly', 'depth', 'minus1depth', 'minus2depth', 'minus3depth', 'minus4depth', 'minus1depth_count', 'switch_mode'])
parser.add_argument("--nb_epochs", type=int, required=True)
parser.add_argument("--save_to", help="Save model to filename")

parser.add_argument("--seed", type=int, help="Set random seed", default=8)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)
parser.add_argument("--test_gates", action="store_true", help="Run diagnostic classifier on gates instead of hidden layer activations")

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--dropout", help="Set dropout fraction", default=0.0)
parser.add_argument("-b", "--batch_size", help="Set batch size", default=24)
parser.add_argument("--val_split", help="Set validation split", default=0.1)

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))
parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=2)
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")
parser.add_argument("--target_folder", help="Set folder to store models", default="dc_models/")

args = parser.parse_args()

####################################################
# Set some params
languages_train             = treebank(seed=args.seed, kind='train')
languages_val              = treebank(seed=args.seed, kind='heldout')
languages_test              = [(name, tb) for name, tb in treebank(seed=args.seed_test, kind='test')]

results_all = {}

if args.test_gates:
    DC = DCgates
else:
    DC = DiagnosticClassifier

training_data = None
validation_data = None

for model in args.models:

    print("\nTraining diagnostic classifier for model %s " % model)
    save_to = args.target_folder+model[:-3]+'_dc'+str(args.seed)

    # find format (for now assume it is in the title) and assure it is right
    format = re.search('postfix|prefix|infix', model).group(0)
    assert format == args.format


    # find recurrent layer
    layer_type = re.search('SimpleRNN|GRU|LSTM', model).group(0)

    training = DC(digits=digits, operators=operators, model=model, classifiers=args.classifiers)

    training_data = training_data or training.generate_training_data(languages_train, format=format)
    validation_data = validation_data or training.generate_training_data(languages_val, format=format)

    training.train(training_data=training_data, validation_data=validation_data, 
            validation_split=args.val_split, batch_size=args.batch_size,
            epochs=args.nb_epochs, verbosity=args.verbosity, filename=save_to,
            save_every=False)


    ######################################################################################
    # Test model and write to file

    # generate_test_data
    test_data = training.generate_test_data(data=languages_test, digits=digits, format=args.format)

    evaluation = training.test(test_data)
    eval_str = training.evaluation_string(evaluation)

    results_all[model[:-3]] = evaluation


# dump all results
pickle.dump(results_all, open(args.target_folder+format+'_'+layer_type+'_dc'+str(args.seed)+'.results','wb'))

