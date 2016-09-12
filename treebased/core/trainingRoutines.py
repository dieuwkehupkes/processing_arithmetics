from __future__ import division
import random
try: import cPickle as pickle
except: import pickle
from multiprocessing import Process, Queue, Pool, Manager
import sys, os, pickle
from collections import defaultdict, Counter

class Treebank():
  def __init__(self,fileList,maxArity):
    self.files = fileList
    self.reset()
    self.maxArity = maxArity
  def getExamples(self):
    if len(self.files) == 1: aFile = self.files[0]
    else:
      try: aFile = self.it.next()
      except:
        self.reset()
        aFile = self.it.next()
    with open(aFile,'rb') as f:
      examples = pickle.load(f)
    examples = [example for example in examples if example.maxArity()<self.maxArity]
    return examples

  def addFiles(self,fileList):
    self.files.extend(fileList)
    self.reset()
  def reset(self):
    random.shuffle(self.files)
    self.it = iter(self.files)

def evaluateBit(theta, testData, q, sample):
  if len(testData)==0:
    q.put(None)
  else:
    try:
      performance = [nw.evaluate(theta,sample) for nw in testData]
      q.put(performance)
    except:
      performance = []
      true = 0
      for (nw,target) in testData:
        performance.append(nw.evaluate(theta,target,sample, True))
        try:
          prediction = nw.predict(theta,None, False,False)
          if prediction == target: true +=1
        except: continue
      print 'accuracy:', true/len(testData)
      q.put(performance)




def evaluate(theta, testData, q = None, description = '', sample=1, cores=1, writeFile=None):
  if len(testData)<1:
    print 'no test examples'
    sys.exit()

  if cores>1:
    myQueue = Queue()
    pPs = []
    bitSize = len(testData)//cores+1
    for i in xrange(cores):
      databit = testData[i*bitSize:(i+1)*bitSize]
      p = Process(name='evaluate', target=evaluateBit, args=(theta, databit, myQueue,sample))
      pPs.append(p)
      p.start()
  else:
    pPs = [0]
    myQueue = Queue()
    evaluateBit(theta, testData, myQueue,sample)


  performance = []
  for p in pPs:
    p = myQueue.get()
    if p is None: continue
    else: performance.extend(p)
  performance = sum(performance)/len(performance)
  if q is None:  return performance
  else:
    confusion = None
    if not writeFile is None:
      with open(writeFile,'a') as f:
        f.write((description,performance))
    q.put((description, performance,confusion))

def phaseZero(tTreebank, vData, hyperParams, theta, cores,outFile):
  if hyperParams['ada']: histGrad = theta.gradient()
  else: histGrad = None
  storeTheta(theta, outFile)
  print '\tStart training'

  print '\tComputing initial performance ('+str(len(vData))+' examples)...'
  est = evaluate(theta, vData, q = None, description = '', sample=0.05, cores=cores)
  print '\tInitial training error: - , Estimated initial performance:', est


  for i in range(hyperParams['startAt'],40,4): # slowy increase sentence length
    examples = tTreebank.getExamples()
    tData = [e for e in examples if e.length()<=i]
    if len(tData)<2:
      print 'skip iteration with sentences up to length',i,'(too few examples)'
      continue
    else: print 'creating training set with sentences up to length',i
    while len(tData)<len(examples):
      tData.extend([e for e in tTreebank.getExamples() if e.length()<=i])
    tData = tData[:len(examples)]

    print '\tIteration with sentences up to length',i,'('+str(len(tData))+' examples)'
    trainLoss=trainOnSet(hyperParams, tData, theta, histGrad, cores)
    storeTheta(theta, outFile)

    print '\tComputing performance ('+str(len(vData))+' examples)...'
    est = evaluate(theta, vData, q = None, description = '', sample=0.05, cores=cores)
    print '\tTraining error:', trainLoss, ', Estimated performance:', est
  print '\tEnd of training phase'

def backAndForth(tTreebank, vData, hyperParams, theta, cores,outFile):
  runs = hyperParams['nEpochs']/10
  hyperParams['nEpochs'] = hyperParams['nEpochs']/10
  hyperParams['fixEmb'] = True
  hyperParams['fixW'] = False

  for run in range(runs):
    print 'run', run,', fixEmb:',hyperParams['fixEmb'], ',fixW:', hyperParams['fixW']
    phase(tTreebank, vData, hyperParams, theta, cores,outFile)
    hyperParams['fixEmb']= not hyperParams['fixEmb']
    hyperParams['fixW']= not hyperParams['fixW']
    if hyperParams['fixEmb']:
      keys = theta.keys()
      keys.remove(('word',))
    elif hyperParams['fixW']:
      keys=[('word',)]
    theta.reset(keys)


def phase(tTreebank, vData, hyperParams, theta, cores,outFile):
  if hyperParams['ada']: histGrad = theta.gradient()
  else: histGrad = None

  storeTheta(theta, outFile)
  print '\tStart training'

  print '\tComputing initial performance ('+str(len(vData))+' examples)...'
  est = evaluate(theta, vData, q = None, description = '', sample=0.05, cores=cores)
  print '\tInitial training error: - , Estimated initial performance:', est


  for i in xrange(hyperParams['nEpochs']):
    tData = tTreebank.getExamples()
    while len(tData)==0: tData = tTreebank.getExamples()
    print '\tIteration',i,'('+str(len(tData))+' examples)'
    trainLoss=trainOnSet(hyperParams, tData, theta, histGrad, cores)
    storeTheta(theta, outFile)
    print '\tComputing performance ('+str(len(vData))+' examples)...'
    est = evaluate(theta, vData, q = None, description = '', sample=0.05, cores=cores)
    print '\tTraining error:', trainLoss, ', Estimated performance:', est
  print '\tEnd of training phase'

def storeTheta(theta, outFile):
  # secure storage: keep back-up of old version until writing is complete
  try: os.rename(outFile, outFile+'.back-up')
  except: True #file did probably not exist, don't bother
  with open(outFile,'wb') as f: pickle.dump(theta,f)
  try: os.remove(outFile+'.back-up')
  except: True #file did not exist, don't bother
  print '\tWrote theta to file: ',outFile

def plainTrain(tTreebank, vTreebank, hyperParams, theta, outDir, cores=1):
  cores = max(1,cores-1)     # 1 for main, 3 for real evaluations, rest for multiprocessing in training and intermediate evaluation
  print 'Using', cores,'core(s) for parallel training and evaluation.'
  print 'Starting plain training'
  outFile = os.path.join(outDir,'plainTrain.theta.pik')

  vData = vTreebank.getExamples()
  #vDataBit = random.sample(vData,int(0.3*len(vData)))

#  backAndForth(tTreebank, vDataBit, hyperParams, theta, cores, outFile)
  phase(tTreebank, vData, hyperParams, theta, cores, outFile)





def beginSmall(tTreebank, vTreebank, hyperParams, theta, outDir, cores=1):
  cores = max(1,cores-1)     # 1 for main
  print 'Using', cores,'cores for parallel training and evaluation.'

  vData = vTreebank.getExamples()
  vDataBit = random.sample(vData,int(0.3*len(vData)))

#  print 'skip phase 0'
  print 'Phase 0: no grammar specialization'
  outFile = os.path.join(outDir,'phase0.theta.pik')
  phaseZero(tTreebank, vDataBit, hyperParams, theta, cores, outFile)

  print 'Phase 1: head specialization'
  theta.specializeHeads()
  outFile = os.path.join(outDir,'phase1.theta.pik')
  phase(tTreebank, vDataBit, hyperParams, theta, cores, outFile)

  print 'Phase 2: rule specialization - most frequent', hyperParams['nRules']
  theta.specializeRules(hyperParams['nRules'])
  outFile = os.path.join(outDir,'phase2.theta.pik')
  phase(tTreebank, vData, hyperParams, adagrad, theta, cores, outFile)


def trainOnSet(hyperParams, examples, theta, histGrad, cores):
  try: fixWords = hyperParams['fixEmb']
  except: fixWords = False
  try: fixWeights = hyperParams['fixW']
  except: fixWeights = False

#  print 'fixEmb:',fixWords, ',fixW:', fixWeights

  adagrad = hyperParams['ada']


  mgr = Manager()
  ns= mgr.Namespace()
  ns.lamb = hyperParams['lambda']
  batchsize = hyperParams['bSize']
  random.shuffle(examples) # randomly split the data into parts of batchsize
  avErrors = []
  for batch in xrange((len(examples)+batchsize-1)//batchsize):
    ns.theta = theta
    minibatch = examples[batch*batchsize:(batch+1)*batchsize]
    s = (len(minibatch)+cores-1)//cores
    trainPs = []
    q = Queue()

    if cores<2:
      trainBatch(ns, minibatch,q, fixWords,fixWeights) #don't start a subprocess
      trainPs.append('')  # But do put a placeholder in the queue
    else:
      for j in xrange(cores):
        p = Process(name='minibatch'+str(batch)+'-'+str(j), target=trainBatch, args=(ns, minibatch[j*s:(j+1)*s],q,fixWords,fixWeights))
        trainPs.append(p)
        p.start()

    errors = []
    theta.regularize(hyperParams['alpha']/len(examples), hyperParams['lambda'])
    for j in xrange(len(trainPs)):
      (grad, error) = q.get()
      if grad is None: continue
      theta.add2Theta(grad,hyperParams['alpha'],histGrad)
      errors.append(error)

    # make sure all worker processes have finished and are killed
    if cores>1:
      for p in trainPs: p.join()

    try: avError = sum(errors)/len(errors)
    except:
      avError = 0
      print 'batch size zero!'
    if batch % 100 == 0:
      print '\t\tBatch', batch, ', average error:',avError , ', theta norm:', theta.norm()
    avErrors.append(avError)
  return sum(avErrors)/len(avErrors)


def trainBatch(ns, examples, q, fixWords = False,fixWeights=False):
  lambdaL2 = ns.lamb
  if len(examples)>0:
    grads = ns.theta.gradient()
    error = 0
    for nw in examples:
      try:
        label = nw[1]
        nw = nw[0]
      except:
        label = None
      derror = nw.train(ns.theta,grads,activate=True, target = label, fixWords=fixWords, fixWeights=fixWeights)
      error+= derror
      if fixWords: grads[('word',)].erase()
    grads /= len(examples)
    q.put((grads, error/len(examples)))
  else:
    q.put((None,None))
