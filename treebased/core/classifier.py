import NN
import numpy as np

class Predictor(NN.Node):
  def __init__(self, child):
    NN.Node.__init__(self, [child], [], ('predict',),'identity')

  def predict(self, theta, activate = True, roundoff = True,verbose = False):
    if activate: self.forward(theta)
    if roundoff: return int(round(self.a[0],0))
    else: return self.a[0]

  def backprop(self, theta, delta, gradient, addOut=False, moveOn=True, fixWords=False, fixWeights=False):
    if fixWeights:  # ignore fixWeights for the classifier weights
      NN.Node.backprop(self, theta, delta, gradient, addOut=addOut, moveOn=False, fixWords=True, fixWeights=False)
    NN.Node.backprop(self, theta, delta, gradient, addOut=addOut, moveOn=moveOn, fixWords=fixWords,
                     fixWeights=fixWeights)

  def error(self, theta, target, activate=True, roundoff = False):
    if activate: self.forward(theta)
    pred = self.predict(theta,activate=activate, roundoff=roundoff)
    return (target-pred)**2

  def train(self, theta, gradient, activate, target, fixWords=False, fixWeights=False):
    if activate: self.forward(theta)
    delta = -2*(target - self.a)
    # TODO: write appropriate delta message
    #delta = None
    self.backprop(theta, delta, gradient, addOut=False, moveOn=True, fixWords=fixWords, fixWeights=fixWeights)

    return self.error(theta, target, False)

  def evaluate(self, theta, target, activate=True):
    return self.error(theta,target,activate)

class Classifier(NN.Node):
  def __init__(self,children, labels, fixed):
    if fixed: children = [NN.Leaf([],('word',),i) for i in range(children)]
    comparison = NN.Node(children, [self], ('comparison',),'ReLU')
    NN.Node.__init__(self,[comparison], [], ('classify',),'softmax')
    self.labels = labels

  def replaceChildren(self,children, fixed):
    if fixed:
      for i in range(len(children)):
        self.inputs[0].inputs[i].key = children[i]
    else: self.inputs[0].inputs = children

  def backprop(self, theta, delta, gradient, addOut = False, moveOn=True, fixWords = False,fixWeights=False):
    if fixWeights: #ignore fixWeights for the classifier weights
      NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=False, fixWords = True,fixWeights=False)
    NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=moveOn, fixWords = fixWords,fixWeights=fixWeights)

  def train(self,theta,gradient,activate, target,fixWords, fixWeights):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] -= 1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True, fixWords = fixWords, fixWeights=fixWeights)
    error = self.error(theta,target,False)
    return error

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except:
      err = -np.log(1e-10)
    return err

  def evaluate(self, theta, target, sample=1, verbose=False):
    return self.error(theta,target)

  def evaluate2(self, theta, children, gold, fixed = True):
    self.replaceChildren(children, fixed)
    loss = self.error(theta,gold,True)
    return loss

  def predict(self,theta,children=None, fixed = True, activate = True, verbose = False):
    if children is not None: self.replaceChildren(children, fixed)
    if activate: self.forward(theta)
    return self.labels[self.a.argmax(axis=0)]

  def __str__(self):
    return 'classify: '+', '.join([str(ch) for ch in self.inputs[0].inputs])



class ClassifierNoComparison(NN.Node):
  def __init__(self,children, labels, fixed):
    if fixed: children = [NN.Leaf([],('word',),i) for i in range(children)]
    NN.Node.__init__(self,children, [], ('classify',),'softmax')
    self.labels = labels

  def replaceChildren(self,children, fixed):
    if fixed:
      for i in range(len(children)):
        self.inputs[i].key = children[i]
    else: self.inputs = children

  def backprop(self, theta, delta, gradient, addOut = False, moveOn=True, fixWords = False,fixWeights=False):
    if fixWeights: #ignore fixWeights for the classifier weights
      NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=False, fixWords = True,fixWeights=False)
    NN.Node.backprop(self,theta, delta, gradient, addOut = addOut, moveOn=moveOn, fixWords = fixWords,fixWeights=fixWeights)

  def train(self,theta,gradient,activate, target,fixWords, fixWeights):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] -= 1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True, fixWords = fixWords, fixWeights=fixWeights)
    error = self.error(theta,target,False)
    return error

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except:
      err = -np.log(1e-10)
    return err

  def evaluate(self, theta, target, sample=1, verbose=False):
    return self.error(theta,target,True)

  def evaluate2(self, theta, children, gold, fixed = True):
    self.replaceChildren(children, fixed)
    loss = self.error(theta,gold,True)
    return loss

  def predict(self,theta,children=None, fixed = True, activate = True, verbose = False):
    if children is not None: self.replaceChildren(children, fixed)
    if activate: self.forward(theta)
    return self.labels[self.a.argmax(axis=0)]

  def __str__(self):
    return 'classify: '+', '.join([str(ch) for ch in self.inputs[0].inputs])


