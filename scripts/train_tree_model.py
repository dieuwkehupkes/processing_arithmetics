from __future__ import division

from processing_arithmetics.tree_based import data, myTheta
import pickle
from processing_arithmetics.tree_based import training_routines as ctr
from processing_arithmetics.tree_based import prediction_training as ptr
from processing_arithmetics.arithmetics import treebanks
import argparse
import os

def main(args):
    # initialize theta (object with model parameters)
    theta = myTheta.install_theta(args['pars_c'],seed=args['seed'],d=(args['dim'],args['dword']),comparison=args['comparison'])

    # generate training and heldout data for comparion training and train model
    dataset_c = data.data4comparison(seed=args['seed'], comparisonLayer=args['comparison'],debug=args['debug'])
    comparison_args={k[:-1]:v for (k,v) in args.iteritems() if k[-1]=='c'}
    comparison_args['out_dir']=args['out_dir']
    print('Comparison training:'+ str(comparison_args))
    ctr.train_comparison(comparison_args, theta, dataset_c)

    # generate training and heldout data for prediction training and train model
    dataset_p = data.data4prediction(theta, seed=args['seed'], debug=args['debug'])
    prediction_args = {k[:-1]: v for (k, v) in args.iteritems() if k[-1] == 'p'}
    prediction_args['out_dir'] = args['out_dir']
    print('Prediction training:' + str(prediction_args))
    ptr.train_prediction(prediction_args,dataset_p,'prediction')


def mybool(string):
    if string in ['F', 'f', 'false', 'False']: return False
    elif string in ['T', 't', 'true', 'True']: return True
    else: raise Exception('Not a valid choice for arg: '+string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('-debug','--debug',type=mybool, default=False, required=False)
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed to be used', required=False)
    # storage:
    parser.add_argument('-o','--out_dir', type=str, help='Output dir to store models', required=True)
    parser.add_argument('-pc','--pars_c', type=str, default='', help='Existing model file (TreeRNN)', required=False)
    parser.add_argument('-pp', '--pars_p', type=str, default='', help='Existing model file (Keras)', required=False)
    # network hyperparameters TreeRNN:
    parser.add_argument('-dc','--comparison', type=int, default=0, help='Dimensionality of comparison layer (0 is no layer)', required=False)
    parser.add_argument('-d','--dim', type=int, default = 2, help='Dimensionality of internal representations', required=False)
    parser.add_argument('-dw','--dword', type=int, default = 2, help='Dimensionality of word embeddings', required=False)
    # network hyperparameters Prediction:
    parser.add_argument('-dh', '--d_hiddenP', type=int, default=0, help='Dimensionality of hidden layer (0 is no layer)', required=False)
    parser.add_argument('-loss', '--loss_p',choices=['mse', 'mae'], default='mse', help='Loss function for prediction', required=False)
    # training hyperparameters comparison:
    parser.add_argument('-opt_c', '--optimizer_c', type=str, default='sgd', choices=['sgd', 'adagrad', 'adam'], help='Optimization scheme for comparison training', required=False)
    parser.add_argument('-nc','--n_epochsC', type=int, default=100, help='Number of epochs for comparison training', required=False)
    parser.add_argument('-bc','--b_sizeC', type=int, default = 50, help='Batch size for comparison training', required=False)
    parser.add_argument('-f', '--storage_freqC', type=int, default=10, help='Model is evaluated and stored after every f epochs', required=False)
    parser.add_argument('-lc','--lambda_c', type=float, default=0.0001, help='Regularization parameter lambda_l2', required=False)
    parser.add_argument('-lrc','--learningRateC', type=float, default=0.01, help='Learning rate parameter', required=False)

    # training hyperparameters prediction:
    parser.add_argument('-np', '--n_epochsP', type=int, default=100, help='Number of epochs for prediction training',
                        required=False)
    parser.add_argument('-bp', '--b_sizeP', type=int, default=50, help='Batch size for prediction training', required=False)
    parser.add_argument('-lp','--lambda_p', type=float, default=0.0001, help='Regularization parameter lambda_l2', required=False)
    parser.add_argument('-lrp','--learningRateP', type=float, default=0.01, help='Learning rate parameter', required=False)

    args = vars(parser.parse_args())

    main(args)

