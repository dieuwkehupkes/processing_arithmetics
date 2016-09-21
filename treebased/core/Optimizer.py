import numpy as np

class Optimizer():
    def __init__(self, theta, lr = 0.01, lambdaL2 = 0):
        self.theta = theta
        self.lr = lr
        self.lambdaL2 = lambdaL2

    def newGradient(self):
        return self.theta.gradient()

## regularization
## TODO: check if this is correct (mostly the 'amount')
    def regularize(self, portion, tofix = []):
        if self.lambdaL2 ==0: return
        for name in self.theta.keys():
            if name[0] in tofix: continue
            if name[-1] == 'M':
                self[name] = (1 - portion * self.lambdaL2) * self[name]
            else:
                continue


class SGD(Optimizer):
    def __init__(self, theta, lr = 0.01, lambdaL2 = 0):
        Optimizer.__init__(self,theta, lr=lr, lambdaL2=lambdaL2)

    def update(self, grads, portion, tofix=[]):
        lr = portion * self.lr
        self.regularize(lr, tofix)

        for key, grad in grads.iteritems():
            if key[0] in tofix:
                continue
            else:
                self.theta[key] -= lr * grad


class Adagrad(Optimizer):
    def __init__(self, theta, lr=0.01, epsilon = 1e-8):
        Optimizer.__init__(self,theta,lr=lr)
        self.histgrad = self.theta.gradient()
        self.epsilon = epsilon

    def update(self, grads, portion, tofix=[]):
        lr = portion * self.lr
        self.regularize(lr, tofix)

        for key, grad in grads.iteritems():
            if key[0] in tofix:
                continue
            else:
                if type(grad) == np.ndarray:
                    self.theta[key] -= lr * np.divide(grad, np.sqrt(self.histgrad[key]) + self.epsilon)
                    self.histgrad[key] += np.square(grad)
                elif grad == WordMatrix:
                    for word in grad:
                        self[key][word] -= lr * np.divide(grad[word], np.sqrt(self.histgrad[key][word]) + self.epsilon)
                        self.histgrad[word] += np.square(grad[word])
                else:
                    raise NameError("Cannot update theta")








