from __future__ import division
from numpy import random as np_random
from matplotlib import pyplot as plt
import Optimizer
import random as random0
import os
try: import cPickle as pickle
except: import pickle
from collections import defaultdict

'''
Storage of parameters as pickled theta object
'''
def store_theta(theta, out_file):
    # secure storage: keep back-up of old version until writing is complete
    try: os.rename(out_file, out_file+'.back-up')
    except: True #file did probably not exist, don't bother

    with open(out_file,'wb') as f: pickle.dump(theta,f)

    try: os.remove(out_file+'.back-up')
    except: True #file did not exist, don't bother
    print '\tWrote theta to file: ',out_file

def train_comparison(args, theta, dataset):
    hyper_params = {k:args[k] for k in ['b_size']}
    hyper_params['to_fix'] = [] # a selection of parameters can be fixed, e.g. the word embeddings
    # initialize optimizer with learning rate (other hyperparams: default values)
    opt = args['optimizer']
    if opt == 'adagrad': optimizer = Optimizer.Adagrad(theta,lr = args['learningrate'], lambda_l2=args['lambda'])
    elif opt == 'adam': optimizer = Optimizer.Adam(theta,lr = args['learningrate'], lambda_l2=args['lambda'])
    elif opt == 'sgd': optimizer = Optimizer.SGD(theta,lr = args['learningrate'], lambda_l2=args['lambda'])
    else: raise RuntimeError("No valid optimizer chosen")

    # train model
    evals = plain_train(optimizer, dataset, hyper_params, n_epochs=args['n_epochs'], outdir=args['out_dir'])

    # store learned model
    store_theta(theta, os.path.join(args['out_dir'], 'comparisonFinalModel.theta.pik'))

    # run final evaluation
    for name, tb in dataset.iteritems():
        print('Evaluation on '+name+' data ('+str(len(tb.examples))+' examples)')
        tb.evaluate(optimizer.theta, verbose=1)


    # create convergence plot
    for name, eval in evals.items():
        toplot = [e[key] for e in eval for key in e if 'loss' in key]
        plt.plot(xrange(len(toplot)), toplot,label=name)
    plt.legend()
    plt.title([key for key in eval[0].keys() if 'loss' in key][0])
    plt.savefig(os.path.join(args['out_dir'],'convergenceComparisonTraining.png'))

'''
Train for nEpochs epochs on tTreebank.
Evaluate on hTreebank after each epoch
Print out traindata performance every 'verbose' batches
'''
def plain_train(optimizer, dataset, hyper_params, n_epochs, verbose=50,f=10, outdir='tmp'):
    batchsize = hyper_params['b_size']
    evals = defaultdict(list)
    t_data = dataset['train'].get_examples()

    for i in range(n_epochs):
        # store model parameters,
        # train f epochs and run an evaluation
        if i%f ==0: # every f epochs: store model parameters and do verbose evaluation
            out_file = os.path.join(outdir, 'comparisonStartEpoch' + str(i * f) + '.theta.pik')
            store_theta(optimizer.theta, out_file)
            for name, tb in dataset.iteritems():
                print('Evaluation on ' + name + ' data')
                tb.evaluate(optimizer.theta, verbose=1)

        print 'Epoch', i, '(' + str(len(t_data)) + ' examples)'

        np_random.shuffle(t_data)  # randomly split the data into parts of batchsize
        for batch in xrange((len(t_data) + batchsize - 1) // batchsize):
            minibatch = t_data[batch * batchsize:(batch + 1) * batchsize]
            error,grads = train_batch(optimizer.theta, minibatch, to_fix=hyper_params['to_fix'])
            if verbose>0 and batch % verbose == 0:
                print ('\tBatch '+str(batch)+', average error: '+str(error / len(minibatch))+', theta norm: '+str(optimizer.theta.norm()))
            optimizer.update(grads)
        optimizer.regularize()

        for kind in ['train','heldout']:
            eval = dataset[kind].evaluate(optimizer.theta,n=5000, verbose = 0)
            print('\tEstimated ' + ', '.join(([kind + ' ' + metric + ': ' + str(round(value,5)) for (metric, value) in eval.iteritems()])))
            evals[kind].append(eval)
    return evals

'''
Train a single (mini)batch, fix the parameters specified in to_fix
'''
def train_batch(theta, examples, to_fix):
    error = 0
    grads = theta.gradient()
    if len(examples) > 0:
        for nw in examples:
            try:
                label = nw[1]
                nw = nw[0]
            except:
                label = None
            derror = nw.train(theta,grads,activate=True, target = label)
            error += derror
    grads.remove_all(to_fix)
    return error, grads
