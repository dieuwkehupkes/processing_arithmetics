from __future__ import division
#import random
from numpy import random as random
import sys, os, pickle
try: import cPickle as pickle
except: import pickle


def evaluate(theta, testData, sample=1):
  performance = []
  true = 0
  n=0
  examples = random.sample(testData,int(sample*len(testData)))
  for (nw, target) in examples:
    performance.append(nw.evaluate(theta, target, True))
    try:
      prediction = nw.predict(theta, activate = False, verbose = 1)
      if str(prediction) == str(target): true += 1
      n+=1
    except:  continue
  return sum(performance)/len(performance), true/len(examples)


def storeTheta(theta, outFile):
  # secure storage: keep back-up of old version until writing is complete
  try: os.rename(outFile, outFile+'.back-up')
  except: True #file did probably not exist, don't bother
  with open(outFile,'wb') as f: pickle.dump(theta,f)
  try: os.remove(outFile+'.back-up')
  except: True #file did not exist, don't bother
  print 'Wrote theta to file: ',outFile

def alternate(optimizer, outDir, datasets, hyperParams, alt, n=5, names=None,verbose=1, seed = -1):
  if seed == -1:
    print 'Always give theta a seed!'
    sys.exit()
  else:
    np.random.seed(seed)

  if names is not None: assert len(names)==len(alt)
  else: names=['']*len(alt)
  assert len(datasets) == len(alt)
  assert len(hyperParams) == len(alt)

  counters = [0]*len(alt)
  outFile = os.path.join(outDir, 'initial' + '.theta.pik')
  storeTheta(optimizer.theta, outFile)

  for iteration in range(n):
    for phase, dataset in enumerate(datasets):
      tofix = hyperParams[phase]['tofix']
      print 'Training phase', names[phase], iteration, 'fixing:',tofix
      plainTrain(optimizer, dataset['train'],dataset['heldout'], hyperParams[phase], nEpochs=alt[phase], nStart=counters[phase], tofix = tofix)
      outFile = os.path.join(outDir,'phase'+str(phase)+'startEpoch'+str(counters[phase])+'.theta.pik')
      storeTheta(optimizer.theta, outFile)
      for ephase, edataset in enumerate(datasets):
        print 'Evaluation phase', names[ephase]
        for kind, tb in edataset.iteritems():
          if kind=='test': continue
          else: tb.evaluate(optimizer.theta, name=kind,verbose=verbose)
      counters[phase]+=alt[phase]



def plainTrain(optimizer, tTreebank, hTreebank, hyperParams, nEpochs, nStart = 0, tofix = []):
  hData = hTreebank.getExamples()
  tData = tTreebank.getExamples()

  batchsize = hyperParams['bSize']

  for i in xrange(nStart, nStart+nEpochs):

    print '\tEpoch', i, '(' + str(len(tData)) + ' examples)'
    random.shuffle(tData)  # randomly split the data into parts of batchsize
    errors = []
    for batch in xrange((len(tData) + batchsize - 1) // batchsize):
      minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
      grads, error = trainBatch(optimizer.theta, minibatch, tofix = hyperParams['tofix'])
      errors.append(error)
      if batch % 50 == 0:
        print '\t\tBatch', batch, ', average error:', error / len(minibatch)
      optimizer.update(grads, portion=len(minibatch)/len(tData), tofix = tofix)
    loss, acc = evaluate(optimizer.theta,hData)
    print '\tTraining loss:', sum(errors)/len(tData), 'heldout loss:',loss, 'heldout accuracy:',acc


def trainBatch(theta, examples, tofix=[]):
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
  return grads,error
