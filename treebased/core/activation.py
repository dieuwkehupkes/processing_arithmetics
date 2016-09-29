import numpy as np

def activate(vector, nonlinearity):
  if nonlinearity =='identity':
    act = vector
    der = np.ones_like(act)
  elif nonlinearity =='tanh':
    act = np.tanh(vector)
    der = 1- np.square(act)
  elif nonlinearity =='ReLU':
    act = np.array([max(x,0)+0.01*min(x,0) for x in vector])
    der = np.array([1*(x>0) +0.01*(x<0)for x in vector])
  elif nonlinearity =='sigmoid':
    act = 1/(1+np.exp(-1*vector))
    der = act * (1 - act)
  elif nonlinearity =='softmax':
    e = np.exp(vector)
    act = e/np.sum(e)
    der = None #np.ones_like(act)#this is never used
  else:
    print 'no familiar nonlinearity:', nonlinearity,'. Used identity.'
    act, der = activate(vector, 'identity')
  return act, der
