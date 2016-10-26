import NN
import numpy as np

class Predictor(NN.Node):
  def __init__(self, child):
    NN.Node.__init__(self, [child.root], [], ('predict',),'identity')
    self.length = child.length


  def predict(self, theta, activate = True, roundoff = True,verbose = False):
    if activate: self.forward(theta)
    if roundoff: return int(round(self.a[0],0))
    else: return self.a[0]

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

  def evaluate(self, theta, target, activate=True, roundoff = False):
    return self.error(theta,target,activate, roundoff=roundoff)

class Classifier(NN.Node):
  def __init__(self,children, labels, comparison):
    self.comparison = comparison
    if comparison:
      comparisonlayer = NN.Node(children, [self], ('comparison',),'ReLU')
      NN.Node.__init__(self,[comparisonlayer], [], ('classify',),'softmax')
    else: NN.Node.__init__(self,children, [], ('classify',),'softmax')
    self.labels = labels

  def train(self,theta,gradient,activate, target):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] -= 1
    self.backprop(theta, delta, gradient, addOut = False, moveOn=True)
    error = self.error(theta,target,False)
    return error

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    try: err= -np.log(self.a[self.labels.index(target)])
    except:
      err = -np.log(1e-10)
    return err

  def evaluate(self, theta, target, activate = True):
    if activate: self.forward(theta)
    return self.error(theta,target)

  def predict(self,theta,children=None, fixed = True, activate = True, verbose = False):
    if children is not None: self.replaceChildren(children, fixed)
    if activate: self.forward(theta)
    return self.labels[self.a.argmax(axis=0)]

  def __str__(self):
    if self.comparison: return 'classify: '+', '.join([str(ch) for ch in self.inputs[0].inputs])
    else: return 'classify: '+', '.join([str(ch) for ch in self.inputs])
