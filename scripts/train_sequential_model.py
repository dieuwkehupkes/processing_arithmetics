import argparse
from keras.layers import SimpleRNN, GRU, LSTM
from keras.models import load_model
import pickle
import numpy as np
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.sequential.architectures import Training, ScalarPrediction, ComparisonTraining, Seq2Seq, DiagnosticTrainer
from processing_arithmetics.arithmetics.treebanks import treebank
from argument_transformation import get_architecture, get_hidden_layer, max_length
import re
import os

"""
Train a model with the default train/test and validation set
"""


###################################################
# Set some params
digits = np.arange(-10, 11)
operators = ['+', '-']
input_size = 2


###################################################
# Create argument parser

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("-architecture", type=get_architecture, help="Type of architecture used during training: scalar prediction, comparison training, seq2seq or a diagnostic classifier", choices=[ScalarPrediction, ComparisonTraining, Seq2Seq, DiagnosticTrainer], required=True)
parser.add_argument("--hidden", required=True, type=get_hidden_layer, help="Hidden layer type", choices=[SimpleRNN, GRU, LSTM])
parser.add_argument("--nb_epochs", required=True, type=int, help="Number of epochs")
parser.add_argument("--save_to", required=True, help="Save trained model to filename")
parser.add_argument("-N", type=int, help="Run script N times", default=1)

# optional arguments
parser.add_argument("-targets", nargs="*", choices=['subtracting', 'intermediate_locally', 'intermediate_recursively', 'grammatical', 'intermediate_directly', 'depth', 'minus1depth', 'minus2depth', 'minus3depth', 'minus4depth', 'switch_mode'])
parser.add_argument("-size_hidden", type=int, help="Size of the hidden layer", default=15)
parser.add_argument("--seed", type=int, help="Set random seed", default=0)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)
parser.add_argument("--recurrent_activation", help="Activation function for recurrent layer", default='tanh')

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--loss_function", "-loss", help="Loss for training", choices=['mae', 'mse'], default='mse')
parser.add_argument("--loss_weights", help="Dictionary with loss weights for seq2seq training", type=eval)
parser.add_argument("--dropout", help="Set dropout fraction", default=0.0)
parser.add_argument("-b", "--batch_size", help="Set batch size", default=24)
parser.add_argument("--val_split", help="Set validation split", default=0.1)
parser.add_argument("--test", action="store_true", help="Test model after training")

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))

# pretrained model argument
parser.add_argument("-m", "--model", type=str, help="Add pretrained model")
parser.add_argument("-fix_embeddings", action="store_true", help="Fix embedding weights during training")
parser.add_argument("-fix_classifier_weights", action="store_true", help="Fix classifier weights during training")
parser.add_argument("-fix_recurrent_weights", action="store_true", help="Fix recurrent weights during training")

parser.add_argument("--remove", action="store_true", help="Remove stored model after training")
parser.add_argument("--verbosity", "-v", type=int, choices=[0,1,2])
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")
parser.add_argument("--visualise_embeddings", action="store_true", help="Visualise embeddings after training")

#######################################################
# Parse arguments and perform some basic checks

args = parser.parse_args()
if args.architecture == DiagnosticTrainer and args.targets is None:
    parser.error("DiagnosticTrainer requires at least one target")

#######################################################
# create languages

languages_test = [(name, tb) for name, tb in treebank(seed=args.seed_test, kind='test',debug=args.debug)]

#################################################################
# Train model N times and store evaluation results

eval_filename = args.save_to+'_evaluation'
eval_file = open(eval_filename, 'w')
results_all = dict()


training = args.architecture(digits=digits, operators=operators, classifiers=args.targets)

for seed in xrange(args.seed, args.seed+args.N):

    print("\nTrain model for seed %i" % seed)
    
    save_to = args.save_to + '_' + str(seed)

    languages_train = treebank(seed=seed, kind='train', debug=args.debug)
    languages_val = treebank(seed=seed, kind='heldout', debug=args.debug)

    training.generate_model(args.hidden, input_size=input_size,

        input_length=args.maxlen, size_hidden=args.size_hidden,
        fix_classifier_weights=args.fix_classifier_weights, 
        fix_embeddings=args.fix_embeddings,
        fix_recurrent_weights=args.fix_recurrent_weights,
        dropout_recurrent=args.dropout,
        classifiers=args.targets)

    # train model
    training_data = training.generate_training_data(data=languages_train, format=args.format) 
    validation_data = training.generate_training_data(data=languages_val, format=args.format) 

    training.train(training_data=training_data, validation_data=validation_data,
            validation_split=args.val_split, batch_size=args.batch_size,
            optimizer=args.optimizer, loss_functions=args.loss_function,
            epochs=args.nb_epochs, verbosity=args.verbosity, filename=save_to,
            save_every=False, visualise_embeddings=args.visualise_embeddings,
            loss_weights=args.loss_weights)

    print("Save model")
    hist = training.trainings_history
    history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
    if not args.remove:
        pickle.dump(history, open(save_to + '.history', 'wb'))
        if args.visualise_embeddings:
            pickle.dump(training.embeddings_anim, open(save_to + '.anim', 'wb'))


    ###############################################################################
    # Test model and write to file

    if not args.test:
        os.remove(save_to+'.h5')
        exit()


    # If model is trained in Seq2Seq mode, recreate model as normal ScalarPrediction model
    if args.architecture == 'Seq2Seq' or args.architecture == 'DiagnosticTrainer':
        model = training.model
        training = ScalarPrediction()
        training.add_pretrained_model(training.model)

    # generate test data
    test_data = training.generate_test_data(data=languages_test, digits=digits, format=args.format) 

    # Helper function to print settings to file
    def sum_settings(args):
        # create string of settings
        settings_str = ''
        settings_str += '\n\n\nTrainings architecture: %s' % args.architecture
        settings_str += '\nRecurrent layer: %s' % args.hidden
        settings_str += '\nSize hidden layer: %i' % args.size_hidden
        settings_str += '\nFormat: %s' % args.format
        settings_str += '\nTrain seed: %s' % args.seed
        settings_str += '\nTest seed: %s' % args.seed_test
        settings_str += '\nBatch size: %i' % args.batch_size
        settings_str += '\nNumber of epochs: %i' % args.nb_epochs
        settings_str += '\nOptimizer: %s\n\n' % args.optimizer

        return settings_str

    eval_file.write(sum_settings(args))

    print("Test model")

    results = training.test(test_data)
    results_str = training.evaluation_string(results) 
    eval_file.write('\t'+results_str)

    results_all[save_to] = results

eval_file.close

pickle.dump(results_all, open(args.save_to+'.results', 'wb'))

print("Finished")

