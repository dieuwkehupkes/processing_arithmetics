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
parser.add_argument("--nb_epochs", required=True, type=int, help="Number of epochs")
parser.add_argument("--save_to", required=True, help="Save trained model to filename")

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

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))

# pretrained model argument
parser.add_argument("-m", "--model", type=str, help="Add pretrained model")
parser.add_argument("-fix_embeddings", action="store_true", help="Fix embedding weights during training")
parser.add_argument("-fix_classifier_weights", action="store_true", help="Fix classifier weights during training")
parser.add_argument("-fix_recurrent_weights", action="store_true", help="Fix recurrent weights during training")

parser.add_argument("--remove", action="store_true", help="Remove stored model after training")
parser.add_argument("--verbosity", "-v", type=int, choices=[0,1,2])
parser.add_argument("--debug", action="store_true", help="Run with small treebank for debugging")
parser.add_argument("--plot_embeddings", action="store_true", help="Plot embeddings after training")

args = parser.parse_args()

languages_train = treebank(seed=args.seed, kind='train', debug=args.debug)
languages_val = treebank(seed=args.seed, kind='heldout', debug=args.debug)
languages_test = [(name, treebank) for name, treebank in treebank(seed=args.seed_test, kind='test',debug=args.debug)]


#################################################################
# Train model

training = args.architecture(digits=digits, operators=operators)

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
        epochs=args.nb_epochs, verbosity=args.verbosity, filename=args.save_to,
        save_every=False, plot_embeddings=args.plot_embeddings)

hist = training.trainings_history
history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
if not args.remove:
    pickle.dump(history, open(args.save_to + '.history', 'wb'))


######################################################################################
# Test model and write to file

eval_filename = args.save_to+'_evaluation'
eval_file = open(eval_filename, 'w')

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
    settings_str += '\nOptimizer: %s' % args.optimizer

    return settings_str

eval_file.write(sum_settings(args))

for name, X, Y in test_data:
    acc = training.model.evaluate(X, Y)
    eval_file.write(name)
    print "Accuracy for \n%s:\t\t" % name,
    results = '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(1, len(acc))])
    print results
    eval_file.write('\n'+results)

eval_file.close

os.remove(eval_filename)
os.remove(args.save_to+'.h5')
