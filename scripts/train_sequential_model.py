import argparse
from keras.layers import SimpleRNN, GRU, LSTM
from keras.models import load_model
import pickle
import numpy as np
from processing_arithmetics.arithmetics import MathTreebank
from processing_arithmetics.sequential.architectures import Training, ScalarPrediction, ComparisonTraining, Seq2Seq
from processing_arithmetics.arithmetics.treebanks import treebank
from argument_transformation import get_architecture, get_hidden_layer, max_length
import re
import os

"""
Train a model with the default train/test and validation set
"""

#TODO write better help function


###################################################
# Set some params
digits = np.arange(-10, 11)
operators = ['+', '-']
input_size = 2


###################################################
# Create argument parser

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("-architecture", type=get_architecture, help="Type of architecture used during training: scalar prediction, comparison training, seq2seq or a diagnostic classifier", choices=[ScalarPrediction, ComparisonTraining, Seq2Seq], required=True)
parser.add_argument("--hidden", required=True, type=get_hidden_layer, help="Hidden layer type", choices=[SimpleRNN, GRU, LSTM])
parser.add_argument("-nb_epochs", required=True, type=int, help="Number of epochs")
parser.add_argument("--save_to", help="Save trained model to filename")

# optional arguments
parser.add_argument("-size_hidden", type=int, help="Size of the hidden layer", default=15)
parser.add_argument("--seed", type=int, help="Set random seed", default=0)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--loss_function", "-loss", help="Loss for training", choices=['mae', 'mse'], default='mse')
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
parser.add_argument("--verbosity", "-v", type=int, choices=[0,1,2], default=1)
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")

parser.add_argument("-N", type=int, help="Train multiple models, write to single file, incrementing seed from --seed", default=1)

parser.add_argument("--visualise_embeddings", action="store_true", help="Visualise embeddings after training")

args = parser.parse_args()
save_to = args.hidden.__name__+'_'+args.format+'_'

languages_test = [(name, treebank) for name, treebank in treebank(seed=args.seed_test, kind='test',debug=args.debug)]

# Open file to write results
eval_filename = save_to+'evaluation'
eval_file = open(eval_filename, 'w')

# generate training object
training = args.architecture(digits=digits, operators=operators)

# generate test data used for all files
languages_test = [(name, treebank) for name, treebank in test_treebank(seed=args.seed_test)]


#################################################################
# Train model N times and store evaluation results

results_all = {}

for seed in xrange(args.seed, args.seed+args.N):

    print("\nTrain model for seed %i" %seed)

    languages_train = treebank(seed=args.seed, kind='train', debug=args.debug)
    languages_val = treebank(seed=args.seed, kind='heldout', debug=args.debug)
    
    # Add pretrained model if this is given in arguments
    if args.model:
        training.add_pretrained_model(model=args.model, 
             copy_weights=None,                                  #TODO change this too!
             fix_classifier_weights=args.fix_classifier_weights,
             fix_embeddings=args.fix_embeddings,
             fix_recurrent_weights=args.fix_recurrent_weights,
             dropout_recurrent=args.dropout)

    else:
        training.generate_model(args.hidden, input_size=input_size,
            input_length=args.maxlen, size_hidden=args.size_hidden,
            fix_classifier_weights=args.fix_classifier_weights, 
            fix_embeddings=args.fix_embeddings,
            fix_recurrent_weights=args.fix_recurrent_weights,
            dropout_recurrent=args.dropout)


    # train model
    training_data = training.generate_training_data(data=languages_train, format=args.format) 
    validation_data = training.generate_training_data(data=languages_val, format=args.format) 

    training.train(training_data=training_data, validation_data=validation_data,
            validation_split=args.val_split, batch_size=args.batch_size,
            optimizer=args.optimizer, loss_function=args.loss_function,
            epochs=args.nb_epochs, verbosity=args.verbosity, filename=save_to+str(seed),
            save_every=False)

    hist = training.trainings_history
    history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
    if not args.remove:
        pickle.dump(history, open(save_to+str(seed) + '.history', 'wb'))


    ######################################################################################
    # Test model and write to file


    # Helper function to print settings to file
    def sum_settings(args, seed):
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
        settings_str += '\nOptimizer: %s' % args.optimizer

        return settings_str

    eval_file.write(sum_settings(args, seed))

    evaluation = training.test(test_data)
    eval_str = training.evaluation_string(evaluation)
    if args.verbosity > 1:
        print eval_str

    eval_file.write('\n'+eval_str)

    results_all[seed] = evaluation

eval_file.close

# store all results in dictionary
if not args.remove:
    pickle.dump(results_all, open(save_to[:-1]+'.results', 'wb'))

if args.remove:
    os.remove(eval_filename)
    os.remove(args.save_to+'.h5')
