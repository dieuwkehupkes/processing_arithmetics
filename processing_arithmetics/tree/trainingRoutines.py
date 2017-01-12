from __future__ import division
from numpy import random as npRandom
import random as random0
import sys
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


'''
Train for nEpochs epochs on tTreebank.
Evaluate on hTreebank after each epoch
Print out traindata performance every 'verbose' batches
'''
def plainTrain(optimizer, tTreebank, hTreebank, hyperParams, nEpochs, nStart = 0, verbose=50):
    batchsize = hyperParams['bSize']
    evals = defaultdict(list)
    tData = tTreebank.getExamples()

    for i in xrange(nStart, nStart+nEpochs):

        print 'Epoch', i, '(' + str(len(tData)) + ' examples)'

        npRandom.shuffle(tData)  # randomly split the data into parts of batchsize
        for batch in xrange((len(tData) + batchsize - 1) // batchsize):
            minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
            error,grads = trainBatch(optimizer.theta, minibatch, toFix=hyperParams['toFix'])
            if verbose>0 and batch % verbose == 0:
                print ('\tBatch '+str(batch)+', average error: '+str(error / len(minibatch))+', theta norm: '+str(optimizer.theta.norm()))
            optimizer.update(grads,len(minibatch)/len(tData))

        evals['heldout'].append(hTreebank.evaluate(optimizer.theta,verbose = 0))
        evals['train'].append(tTreebank.evaluate(optimizer.theta,verbose = 0))
        for name, eval in evals.iteritems():
            print('\t'+' '.join(([name+' '+metric+': '+str(value) for (metric, value) in eval[-1].items()])))
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
