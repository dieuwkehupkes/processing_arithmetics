import sys 
sys.path.insert(0, '../arithmetics') 
import argparse
from keras.layers import SimpleRNN, GRU, LSTM
import pickle
import numpy as np
from keras.models import load_model
from architectures import Training, A1, A4, Probing, Seq2Seq
from arithmetics import mathTreebank, training_treebank, test_treebank, heldout_treebank
import re

# Train a model with the default train/test and validation set

###################################################
# Helper functions for argument parsing

def get_architecture(architecture):
    arch_dict = {'A1':A1, 'A4':A4, 'DiagnosticClassifier':Probing, 'DC':Probing, 'Seq2Seq':Seq2Seq}
    return arch_dict[architecture]

def get_hidden_layer(hl_name):
    hl_dict = {'SimpleRNN': SimpleRNN, 'SRN': SimpleRNN, 'GRU': GRU, 'LSTM': LSTM}
    return hl_dict[hl_name]

def max_length(N):
    """
    Compute length of arithmetic expression
    with N numeric leaves
    :param N: number of numeric leaves of expression
    """
    l = 4*N-3
    return l


###################################################
# Get arguments

parser = argparse.ArgumentParser()

# positional arguments
parser.add_argument("architecture", type=get_architecture, help="Type of architecture used during training: scalar prediction, comparison training, seq2seq or a diagnostic classifier", choices=[A1, A4, Probing, Seq2Seq])
parser.add_argument("hidden", type=get_hidden_layer, help="Hidden layer type", choices=[SimpleRNN, GRU, LSTM])
parser.add_argument("nb_epochs", type=int, help="Number of epochs")
parser.add_argument("save_to", help="Save trained model to filename")

# optional arguments
parser.add_argument("-size_hidden", type=int, help="Size of the hidden layer", default=15)
parser.add_argument("--seed", type=int, help="Set random seed", default=0)
parser.add_argument("--format", type=str, help="Set formatting of arithmetic expressions", choices=['infix', 'postfix', 'prefix'], default="infix")
parser.add_argument("--seed_test", type=int, help="Set random seed for testset", default=100)

parser.add_argument("--optimizer", help="Set optimizer for training", choices=['adam', 'adagrad', 'adamax', 'adadelta', 'rmsprop', 'sgd'], default='adam')
parser.add_argument("--dropout", help="Set dropout fraction", default=0.0)
parser.add_argument("-b", "--batch_size", help="Set batch size", default=24)
parser.add_argument("--val_split", help="Set validation split", default=0.1)

parser.add_argument("-maxlen", help="Set maximum number of digits in expression that network should be able to parse", type=max_length, default=max_length(15))

# pretrained model argument
parser.add_argument("-m", "--model", type=str, help="Add pretrained model")
parser.add_argument("-fix_embeddings", action="store_true", help="Fix embedding weights during training")
parser.add_argument("-fix_classifier_weights", action="store_true", help="Fix classifier weights during training")
parser.add_argument("-fix_recurrent_weights", action="store_true", help="Fix recurrent weights during training")


# TODO verbosity?
args = parser.parse_args()

languages_train             = training_treebank(seed=args.seed)
languages_val              = heldout_treebank(seed=args.seed)
languages_test              = [(name, treebank) for name, treebank in test_treebank(seed=args.seed_test)]
digits = np.arange(-10, 11)
dmap = pickle.load(open('best_models/dmap', 'rb'))      # TODO change this!
input_dim = len(dmap)+1
input_size = 2


#################################################################
# Train model

training = args.architecture()
training_data = args.architecture.generate_training_data(architecture=training, data=languages_train, dmap=dmap, format=args.format, pad_to=args.maxlen) 
validation_data = args.architecture.generate_training_data(architecture=training, data=languages_val, dmap=dmap, format=args.format, pad_to=args.maxlen) 

# Add pretrained model if this is given in arguments
if args.model:
    model = load_model(args.model)
    training.add_pretrained_model(model=model, 
         dmap=dmap, copy_weights=None,           #TODO change this too!
         fix_classifier_weights=args.fix_classifier_weights,
         fix_embeddings=args.fix_embeddings,
         fix_recurrent_weights=args.fix_recurrent_weights,
         optimizer=args.optimizer,
         dropout_recurrent=args.dropout)

else:
    training.generate_model(args.hidden, input_dim=input_dim, 
        input_size=input_size, input_length=args.maxlen, 
        size_hidden=args.size_hidden, dmap=dmap,
        fix_classifier_weights=args.fix_classifier_weights, 
        fix_embeddings=args.fix_embeddings,
        fix_recurrent_weights=args.fix_recurrent_weights,
        optimizer=args.optimizer,
        dropout_recurrent=args.dropout)


# train model
training.train(training_data=training_data, validation_data=validation_data,
        validation_split=args.val_split, batch_size=args.batch_size,
        epochs=args.nb_epochs, verbosity=1, filename=args.save_to,
        save_every=False)

training.print_accuracies
     
hist = training.trainings_history
history = (hist.losses, hist.val_losses, hist.metrics_train, hist.metrics_val)
pickle.dump(history, open(args.save_to + '.history', 'wb'))


######################################################################################
# Test model

# generate test data
test_data = args.architecture.generate_test_data(architecture=training, data=languages_test, dmap=dmap, digits=np.arange(-10, 11), format=args.format, pad_to=args.maxlen) 

for name, X, Y in test_data:
    acc = training.model.evaluate(X, Y)
    print "Accuracy for for test set %s:" % name,
    print '\t'.join(['%s: %f' % (training.model.metrics_names[i], acc[i]) for i in xrange(len(acc))])


# TODO still use this somewhere?
def print_sum(settings):
    # print summary of training session
    print('Model summary:')
    print('Recurrent layer: %s' % str(settings.recurrent_layer))
    print('Size hidden layer: %i' % settings.size_hidden)
    print('Initialisation embeddings: %s' % settings.encoding)
    print('Size embeddings: %i' % settings.input_size)
    print('Batch size: %i' % settings.batch_size)
    print('Number of epochs: %i' % settings.nb_epoch)
    print('Optimizer: %s' % settings.optimizer)
    print('Trained on:')
    try:
        for language, nr in settings.languages_train.items():
            print('%i sentences from %s' % (nr, language))
    except AttributeError:
        print('Unknown')
