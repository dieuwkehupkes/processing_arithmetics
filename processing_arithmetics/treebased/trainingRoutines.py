from __future__ import division
from numpy import random as npRandom
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
def storeTheta(theta, outFile):
    # secure storage: keep back-up of old version until writing is complete
    try: os.rename(outFile, outFile+'.back-up')
    except: True #file did probably not exist, don't bother

    with open(outFile,'wb') as f: pickle.dump(theta,f)

    try: os.remove(outFile+'.back-up')
    except: True #file did not exist, don't bother
    print '\tWrote theta to file: ',outFile

def trainComparison(args, theta, dataset):
    hyperParams = {k:args[k] for k in ['bSize']}
    hyperParams['toFix'] = [] # a selection of parameters can be fixed, e.g. the word embeddings
    # initialize optimizer with learning rate (other hyperparams: default values)
    opt = args['optimizer']
    if opt == 'adagrad': optimizer = Optimizer.Adagrad(theta,lr = args['learningRate'], lambdaL2=args['lambda'])
    elif opt == 'adam': optimizer = Optimizer.Adam(theta,lr = args['learningRate'], lambdaL2=args['lambda'])
    elif opt == 'sgd': optimizer = Optimizer.SGD(theta,lr = args['learningRate'], lambdaL2=args['lambda'])
    else: raise RuntimeError("No valid optimizer chosen")

    # train model
    evals = plainTrain(optimizer, dataset, hyperParams, nEpochs=args['nEpochs'], outdir=args['outDir'])

    # store learned model
    storeTheta(theta, os.path.join(args['outDir'], 'comparisonFinalModel.theta.pik'))

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
    plt.savefig(os.path.join(args['outDir'],'comparisonConvergence.png'))

'''
Train for nEpochs epochs on tTreebank.
Evaluate on hTreebank after each epoch
Print out traindata performance every 'verbose' batches
'''
def plainTrain(optimizer, dataset, hyperParams, nEpochs, verbose=50,f=10, outdir='tmp'):
    batchsize = hyperParams['bSize']
    evals = defaultdict(list)
    tData = dataset['train'].getExamples()

    for i in range(nEpochs):
        # store model parameters,
        # train f epochs and run an evaluation
        if i%f ==0: # every f epochs: store model parameters and do verbose evaluation
            outFile = os.path.join(outdir, 'comparisonStartEpoch' + str(i * f) + '.theta.pik')
            storeTheta(optimizer.theta, outFile)
            for name, tb in dataset.iteritems():
                print('Evaluation on ' + name + ' data')
                tb.evaluate(optimizer.theta, verbose=1)

        print 'Epoch', i, '(' + str(len(tData)) + ' examples)'

        npRandom.shuffle(tData)  # randomly split the data into parts of batchsize
        for batch in xrange((len(tData) + batchsize - 1) // batchsize):
            minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
            error,grads = trainBatch(optimizer.theta, minibatch, toFix=hyperParams['toFix'])
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
Train a single (mini)batch, fix the parameters specified in toFix
'''
def trainBatch(theta, examples, toFix):
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
    grads.removeAll(toFix)
    return error, grads
