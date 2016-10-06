from __future__ import division
import random
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
      prediction = nw.predict(theta, activate = False, verbose = False)
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
  print '\tWrote theta to file: ',outFile

def alternate(optimizer, datasets, outDir, hyperParams, alt, n=5, names=None):
  if names is not None: assert len(names)==len(alt)
  else: names=['']*len(alt)
  assert len(datasets) == len(alt)
  assert len(hyperParams) == len(alt)
  counters = [0]*len(alt)
  #histGrad = theta.gradient()
  outFile = os.path.join(outDir, 'initial' + '.theta.pik')
  storeTheta(optimizer.theta, outFile)


  for phase, dataset in enumerate(datasets):
    print 'Evaluating',names[phase]
    [tb.evaluate(optimizer.theta, name=kind) for kind, tb in dataset.iteritems()]

  for iteration in range(n):
    for phase, dataset in enumerate(datasets):
      print 'Training phase', names[phase]
      plainTrain(optimizer, dataset['train'],dataset['heldout'], hyperParams[phase], nEpochs=alt[phase], nStart=counters[phase])
      outFile = os.path.join(outDir,'phase'+str(phase)+'startEpoch'+str(counters[phase])+'.theta.pik')
      storeTheta(optimizer.theta, outFile)
      print 'Evaluation phase', names[phase]
      [tb.evaluate(optimizer.theta, name=kind) for kind, tb in dataset.iteritems()]
      counters[phase]+=alt[phase]



def plainTrain(optimizer, tTreebank, hTreebank, hyperParams, nEpochs, nStart = 0):
  print hyperParams
  hData = hTreebank.getExamples()
  tData = tTreebank.getExamples()
  batchsize = hyperParams['bSize']

  for i in xrange(nStart, nStart+nEpochs):
    print '\tEpoch', i, '(' + str(len(tData)) + ' examples)'
    random.shuffle(tData)  # randomly split the data into parts of batchsize
    errors = []
    for batch in xrange((len(tData) + batchsize - 1) // batchsize):
      #grads = theta.gradient()
      minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
      error,grads = trainBatch(optimizer.theta, minibatch, fixWords=False, fixWeights=False)
      errors.append(error)
      if batch % 50 == 0:
        print '\t\tBatch', batch, ', average error:', error / len(minibatch), ', theta norm:', optimizer.theta.norm()
      optimizer.update(grads,len(minibatch)/len(tData))
    loss, acc = evaluate(optimizer.theta,hData)
    print '\tTraining loss:', sum(errors)/len(tData), 'heldout loss:',loss, 'heldout accuracy:',acc


def trainBatch(theta, examples, fixWords = False,fixWeights=False):
  error = 0
  grads = theta.gradient()
  if len(examples)>0:
    for nw in examples:
      try:
        label = nw[1]
        nw = nw[0]
      except:
        label = None
      derror = nw.train(theta,grads,activate=True, target = label, fixWords=fixWords, fixWeights=fixWeights)
      error+= derror
      if fixWords: grads[('word',)].erase()
  return error, grads
