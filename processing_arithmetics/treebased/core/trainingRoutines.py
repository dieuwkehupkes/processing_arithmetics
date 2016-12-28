from __future__ import division
from numpy import random as npRandom
import random as random0
import sys, os, pickle
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

'''
Training in phases:
datasets, hyperParams, alt and names can be lists that belong to diffent tasks
Training will iterate over these lists n times
 the model will be trained for the number of epochs specified in alt on that task
 using the dataset and hyperparams specified
 After each phase, the model will be evaluated
'''
def alternate(optimizer, datasets, outDir, hyperParams, alt, n=5, names=None,seed = 0):
    npRandom.seed(seed)
    random0.seed(seed)
    if names is not None: assert len(names)==len(alt)
    else: names=['']*len(alt)
    assert len(datasets) == len(alt)
    assert len(hyperParams) == len(alt)
    counters = [0]*len(alt)
    storeTheta(optimizer.theta, os.path.join(outDir, 'initial' + '.theta.pik'))
    evals = defaultdict(list)

    for phase, dataset in enumerate(datasets):
        print 'Evaluating',names[phase]
        [tb.evaluate(optimizer.theta, name=kind) for kind, tb in dataset.iteritems()]

    for iteration in range(n):
        for phase, dataset in enumerate(datasets):
            print 'Training phase', names[phase]
            newEval = plainTrain(optimizer, dataset['train'],dataset['heldout'], hyperParams[phase], nEpochs=alt[phase], nStart=counters[phase])
            [evals[name].extend(eval) for name, eval in newEval.items()]

            counters[phase]+=alt[phase]
            outFile = os.path.join(outDir,'phase'+str(phase)+'startEpoch'+str(counters[phase])+'.theta.pik')
            storeTheta(optimizer.theta, outFile)
            print 'Evaluation phase', names[phase]
            [tb.evaluate(optimizer.theta, name=kind,verbose=1) for kind, tb in dataset.iteritems()]
    return evals

'''
Simply train for nEpochs epochs on tTreebank.
Evaluate after each epoch
Print out traindata performance every 50 batches
'''
def plainTrain(optimizer, tTreebank, hTreebank, hyperParams, nEpochs, nStart = 0):
    print hyperParams

    batchsize = hyperParams['bSize']
    evals=defaultdict(list)
    tData = tTreebank.getExamples()

    for i in xrange(nStart, nStart+nEpochs):

        print '\tEpoch', i, '(' + str(len(tData)) + ' examples)'

        npRandom.shuffle(tData)  # randomly split the data into parts of batchsize
        for batch in xrange((len(tData) + batchsize - 1) // batchsize):
            minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
            error,grads = trainBatch(optimizer.theta, minibatch, toFix=hyperParams['toFix'])
            if batch % 50 == 0:
                print '\t\tBatch', batch, ', average error:', error / len(minibatch), ', theta norm:', optimizer.theta.norm()
            optimizer.update(grads,len(minibatch)/len(tData))

        evals['heldout'].append(hTreebank.evaluate(optimizer.theta,verbose = 0))
        evals['train'].append(tTreebank.evaluate(optimizer.theta,verbose = 0))
        for name, eval in evals.iteritems():
            print '\t'+' '.join(([name+' '+metric+': '+str(value) for (metric, value) in eval[-1].items()]))
    return evals

'''
Train a single (mini)batch, fix the parameters specified by toFix
'''
def trainBatch(theta, examples, toFix):
    error = 0
    grads = theta.gradient()
    if len(examples)>0:
        for nw in examples:
            try:
                label = nw[1]
                nw = nw[0]
            except:
                label = None
            derror = nw.train(theta,grads,activate=True, target = label)
            error+= derror
    grads.removeAll(toFix)
    return error, grads
