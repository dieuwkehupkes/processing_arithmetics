import numpy as np

def gradientCheck(theta, network, target=None):
  # compute analyticial and numerical gradient
  print 'computing analytical gradient'
  grad = theta.gradient()
  network.train(theta, grad, activate=True, target=target)
  numgrad = numericalGradient(network, theta, target)

  print 'Comparing analytical to numerical gradient'
  # flatten gradient objects and report difference
  gradflat = np.array([])
  numgradflat = np.array([])
  for name in theta.keys():
    if name == ('word',): #True
      ngr = np.concatenate([numgrad[name][word] for word in theta[name].keys()]) #reshape(numgrad[name],-1)
      gr = np.concatenate([grad[name][word] for word in theta[name].keys()]) #np.reshape(grad[name],-1)

    else:
      ngr = np.reshape(numgrad[name],-1)
      gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+str(name)+' :', diff
    if False:#diff>0.001:
      print '    ','gr\t\tngr\t\td'
      for i in range(len(gr)):
        if gr[i]==0 and ngr[i]==0: v = str(0)
        else: v= str(abs((gr[i]-ngr[i])/(gr[i]+ngr[i])))
        try: print '    ',theta.lookup[name][i//len(grad[name][0])], str(gr[i])+'\t'+str(ngr[i])+'\t'+v
        except: print '    ',str(gr[i])+'\t'+str(ngr[i])+'\t'+v
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))

def numericalGradient(nw, theta, target = None):
#    nw.activateNW(theta)
    print '\n\nComputing numerical gradient for target', target
    epsilon = 0.0001
    numgrad = theta.gradient()

    for name in theta.keys():
      if name == ('word',): #True
        for word in theta[name].keys():
          for i in range(len(theta[name][word])):
            old = theta[name][word][i]
            theta[name][word][i] = old + epsilon
            errorPlus=nw.error(theta,target,True)
            theta[name][word][i] = old - epsilon
            errorMin=nw.error(theta,target,True)
            d =(errorPlus-errorMin)/(2*epsilon)
            numgrad[name][word][i] = d
            theta[name][word][i] = old  # restore theta
      else:
    # create an iterator to iterate over the array, no matter its shape
        it = np.nditer(theta[name], flags=['multi_index'])

        while not it.finished:
          i = it.multi_index
#          print '\n\t',i
          old = theta[name][i]
          theta[name][i] = old + epsilon
          errorPlus=nw.error(theta,target,True)
          theta[name][i] = old - epsilon
          errorMin=nw.error(theta,target,True)
          d =(errorPlus-errorMin)/(2*epsilon)
          numgrad[name][i] = d
          theta[name][i] = old  # restore theta
          it.iternext()
    return numgrad