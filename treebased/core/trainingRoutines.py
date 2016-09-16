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

def alternate(theta, datasets, outDir, hyperParams, alt, n=5, names=None,verbose=False):
  if names is not None: assert len(names)==len(alt)
  else: names=['']*len(alt)
  assert len(datasets) == len(alt)
  assert len(hyperParams) == len(alt)
  counters = [0]*len(alt)
  histGrad = theta.gradient()
  outFile = os.path.join(outDir, 'initial' + '.theta.pik')
  storeTheta(theta, outFile)


  for phase, dataset in enumerate(datasets):
    print 'Evaluating',names[phase]
    [tb.evaluate(theta, name=kind) for kind, tb in dataset.iteritems()]

  for iteration in range(n):
    for phase, dataset in enumerate(datasets):
      print 'Training phase', names[phase]
      plainTrain(dataset['train'],dataset['heldout'], theta, hyperParams[phase], nEpochs=alt[phase], nStart=counters[phase], histGrad=histGrad)
      outFile = os.path.join(outDir,'phase'+str(phase)+'startEpoch'+str(counters[phase])+'.theta.pik')
      storeTheta(theta, outFile)
      for ephase, edataset in enumerate(datasets):
        print 'Evaluation phase', names[ephase]
        [tb.evaluate(theta, name=kind,verbose=verbose) for kind, tb in edataset.iteritems()]
      counters[phase]+=alt[phase]



def plainTrain(tTreebank, hTreebank, theta, hyperParams, nEpochs, nStart = 0, histGrad = None):
  hData = hTreebank.getExamples()
  tData = tTreebank.getExamples()

  if histGrad is None and hyperParams['ada']: histGrad = theta.gradient()

  batchsize = hyperParams['bSize']

  for i in xrange(nStart, nStart+nEpochs):


    print '\tEpoch', i, '(' + str(len(tData)) + ' examples)'
    random.shuffle(tData)  # randomly split the data into parts of batchsize
    errors = []
    for batch in xrange((len(tData) + batchsize - 1) // batchsize):
      grads = theta.gradient()
      minibatch = tData[batch * batchsize:(batch + 1) * batchsize]
      error = trainBatch(theta, grads, minibatch, tofix = hyperParams['tofix'])
      errors.append(error)
      if batch % 50 == 0:
        print '\t\tBatch', batch, ', average error:', error / len(minibatch), ', theta norm:', theta.norm()

      # update theta: regularize and apply collected gradients
      theta.regularize(hyperParams['alpha'] / len(minibatch), hyperParams['lambda'], tofix=hyperParams['tofix'])



      theta.add2Theta(grads, hyperParams['alpha'], histGrad, tofix=hyperParams['tofix'])
    loss, acc = evaluate(theta,hData)
    print '\tTraining loss:', sum(errors)/len(tData), 'heldout loss:',loss, 'heldout accuracy:',acc


def trainBatch(theta, grads, examples, tofix=[]):
  error = 0
  if len(examples)>0:
    #print 'done'

    for nw in examples:
      try:
        label = nw[1]
        nw = nw[0]
      except:
        label = None
      #print 'start training example:', str(nw), label
      derror = nw.train(theta,grads,activate=True, target = label)
      error+= derror
      for name in tofix: grads.erase(name) #[name].erase()

    grads /= len(examples)
  return error
