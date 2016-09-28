import NN
import numpy as np

class Predictor(NN.Node):
  def __init__(self, child, hiddenLayer = False):
    self.hiddenLayer = hiddenLayer


    if hiddenLayer: NN.Node.__init__(self, [NN.Node([child],[],('predictH',),'tanh')], [], ('predict',),'identity')
    else: NN.Node.__init__(self, [child], [], ('predict',),'identity')
    #self.root = child
    self.length = (len(str(child).split())+3)/4

  def predict(self, theta, activate = True, roundoff = True,verbose = False):
    if activate: self.forward(theta)
    if roundoff: return int(round(self.a[0],0))
    else: return self.a[0]

  #def backprop(self, theta, delta, gradient):
  #  NN.Node.backprop(self, theta, delta, gradient)

  def error(self, theta, target, activate=True, roundoff = False):
    if activate: self.forward(theta)
    pred = self.predict(theta,activate=activate, roundoff=roundoff)
    return (target-pred)**2

  def train(self, theta, gradient, activate, target):
    if activate: self.forward(theta)
    #print 'prediction:', self.predict(theta,False,False), ', target:',target, ', error:', self.error(theta, target, False)
    delta = -2*(target - self.a)
    self.backprop(theta, delta, gradient)

    return self.error(theta, target, False)

  def evaluate(self, theta, target, activate=True, roundoff = False):
    return self.error(theta,target,activate, roundoff=roundoff)


class Classifier(NN.Node):
  def __init__(self,children, labels, noComparison = False):
    if not noComparison:
      comparison = NN.Node(children, [self], ('comparison',),'ReLU')
      NN.Node.__init__(self,[comparison], [], ('classify',),'softmax')
    else:
      NN.Node.__init__(self, children, [], ('classify',), 'softmax')
    self.labels = labels

  def train(self,theta,gradient,activate, target):
    if activate: self.forward(theta)
    delta = np.copy(self.a)
    true = self.labels.index(target)
    delta[true] -= 1
    self.backprop(theta, delta, gradient)
    error = self.error(theta,target,False)
    return error

  def error(self,theta, target, activate=True):
    if activate: self.forward(theta)

    err= -np.log(self.a[self.labels.index(target)])
    # except:
    #   print 'would be zero!'
    #   err = -np.log(1e-10)
    return err

  def evaluate(self, theta, target, activate=True):
    return self.error(theta,target,activate)


  def predict(self,theta, activate = True, verbose = False):
    if activate: self.forward(theta)
    return self.labels[self.a.argmax(axis=0)]

  def __str__(self):
    return 'classify: '+', '.join([str(ch) for ch in self.inputs[0].inputs])