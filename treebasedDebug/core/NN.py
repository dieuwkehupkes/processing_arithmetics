from __future__ import division
import activation
import numpy as np
import sys

import myTheta#,myRAE
#from myRAE import Reconstruction


class Node():
  def __init__(self,inputs, outputs, cat,nonlinearity):
#    print 'Node.init', cat, inputs
    self.inputs = inputs
    self.outputs = outputs
    self.cat = cat
    self.nonlin = nonlinearity

  def forward(self,theta, activateIn = True, activateOut = False, signal=None):
#    print self.inputs
#    print 'Node.forward',self.cat# #theta[('composition', '#X#', '(#X#, #X#)', 'I', 'M')][0][:3]
    if activateIn:
      [i.forward(theta, activateIn,activateOut) for i in self.inputs]
    if signal is None:
      self.inputsignal = np.concatenate([c.a for c in self.inputs])
      self.dinputsignal = np.concatenate([c.ad for c in self.inputs])
    else:
      self.inputsignal = signal[0]
      self.dinputsignal = signal[1]

    M= theta[self.cat+('M',)]
    b= theta[self.cat+('B',)]
    if M is None or b is None: raise RuntimeError('Fail to forward node, no matrix and bias vector:'+str(self.cat))
#    print self.cat, M.shape


    try:
      self.z = M.dot(self.inputsignal)+b
      self.a, self.ad = activation.activate(self.z, self.nonlin)
    except:
       print 'problem', self.cat, self.inputsignal.shape, M.shape, b.shape
       self.z = M.dot(self.inputsignal)+b
       self.a, self.ad = activation.activate(self.z, self.nonlin)
    if activateOut:
      for node in self.outputs:
        if node.cat[0]=='reconstruction':
          node.forward(theta, False, False, signal=(self.a,self.ad))
        else:
          node.forward(theta, False, True, signal=None)

  def backprop(self,theta, delta, gradient, addOut = False, moveOn=True, fixWords = False,fixWeights=False):
#    print 'Node.backprop',fixWords, fixWeights #self#.cat#, 'a:', self.a.shape,'delta:', delta.shape, 'input:', self.inputsignal.shape

    if addOut: #add a delta message from its outputs (e.g., reconstructions)
      delta += np.concatenate([out.backprop(theta, None, gradient, addOut, moveOn=False, fixWords = fixWords, fixWeights=fixWeights) for out in self.outputs])
    M= theta[self.cat+('M',)]
    if not fixWeights:
      gradient[self.cat+('M',)]+= np.outer(delta,self.inputsignal)
      gradient[self.cat+('B',)]+=delta

    deltaB =np.multiply(np.transpose(M).dot(delta),self.dinputsignal)
    if moveOn:
      lens = [len(c.a) for c in self.inputs]
      splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
      [inputNode.backprop(theta, delt, gradient,addOut, moveOn, fixWords, fixWeights) for inputNode,delt in zip(self.inputs,np.split(deltaB,splitter))]
    else:
      return deltaB


  def __str__(self):
    return '( '+ ' '.join([str(child) for child in self.inputs])+' )'
#     if self.cat[-1]=='I': return '('+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+' <'+' '.join([str(out) for out in self.outputs])+'>'+')'
#     if self.cat[-1]=='O': return '['+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+']' #'&'.join([str(c) for c in self.inputs])
#     else: return '<'+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+'> '
#     #return '['+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+']'


class Leaf(Node):
  def __init__(self,outputs,cat, key=0,nonlinearity='identity'):
    Node.__init__(self,[], outputs, cat,nonlinearity)
    self.key = key

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'Leaf.forward', self.cat, self.key#, type(self.key)
    try: self.z = theta[self.cat][self.key]
#     try: self.z = theta[self.cat][self.key]
    except:
      print 'Fail to forward Leaf:', self.cat, self.key, type(self.key)
      sys.exit()

    self.a, self.ad = activation.activate(self.z,self.nonlin)
    if activateOut:
      [i.forward(theta, False,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient, addOut = False, moveOn = False, fixWords = False,fixWeights=False):
#    if self.key == 'UNKNOWN': print 'node has an unknown key'
    if not fixWords:
#      print 'update:', self.key
      gradient[self.cat][self.key] += delta

  def aLen(self,theta):
    return len(theta[self.cat][self.key])

  def __str__(self):
    return str(self.key)#+'('+self.cat+')'