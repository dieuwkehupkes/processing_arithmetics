import numpy as np
import myTheta

class Optimizer():
    def __init__(self, theta, lr = 0.01, lambdaL2 = 0.0001):
        self.theta = theta
        self.lr = lr
        self.lambdaL2 = lambdaL2

    def newGradient(self):
        return self.theta.gradient()

# regularization
# TODO: check if this is correct 
    def regularize(self, portion=1, tofix = []):
        if self.lambdaL2 == 0: return
        for name in self.theta.keys():
            if name[0] in tofix: continue
            if name[-1] == 'M':
                self.theta[name] = (1 - portion * self.lambdaL2) * self.theta[name]
            else:
                continue


class SGD(Optimizer):
    def __init__(self, theta, lr = 0.01, lambdaL2 = 0):
        Optimizer.__init__(self,theta, lr=lr, lambdaL2=lambdaL2)

    def update(self, grads, tofix=[]):
        lr =  self.lr

        for key, grad in grads.iteritems():
            if key[0] in tofix:
                continue
            else:
                if type(grad) == np.ndarray: self.theta[key] -= lr * grad
                elif type(grad) == myTheta.WordMatrix:
                    for word in grad: self.theta[key][word] -= lr * grad[word]
                else: raise NameError("Cannot update theta")


class Adagrad(Optimizer):
    def __init__(self, theta, lr=0.01, lambdaL2 = 0, epsilon = 1e-8):
        Optimizer.__init__(self,theta,lr=lr, lambdaL2=lambdaL2)
        self.histgrad = self.theta.gradient()
        self.epsilon = epsilon

    def update(self, grads, tofix=[]):
        lr = self.lr

        for key, grad in grads.iteritems():
            if key[0] in tofix:
                continue
            else:
                if type(grad) == np.ndarray:
                    self.histgrad[key] += np.square(grad)
                    self.theta[key] -= lr * np.divide(grad, np.sqrt(self.histgrad[key]) + self.epsilon)
                elif type(grad) == myTheta.WordMatrix:
                    for word in grad:
                        self.histgrad[key][word] += np.square(grad[word])
                        self.theta[key][word] -= lr * np.divide(grad[word], np.sqrt(self.histgrad[key][word]) + self.epsilon)
                else:
                    raise NameError("Cannot update theta")

class Adam(Optimizer):
    def __init__(self,theta,lr=0.001, lambdaL2=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-8):
        Optimizer.__init__(self,theta,lr,lambdaL2)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = theta.gradient()
        self.vs = theta.gradient()

    def update(self, grads, tofix=[]):
        self.t += 1

        factor = (1 - self.beta_2**self.t)**0.5/(1 - self.beta_1**self.t)
        lr = factor*self.lr

        for key, grad in grads.iteritems():
            if key[0] in tofix:
                continue
            else:
                if type(grad) == np.ndarray:
                    self.ms[key] = self.beta_1*self.ms[key]+(1-self.beta_1)*grad
                    self.vs[key] = self.beta_2 * self.vs[key] + (1 - self.beta_2) * np.square(grad)
                    self.theta[key] -= lr * self.ms[key] / np.sqrt(self.vs[key] + self.epsilon)
                elif type(grad) == myTheta.WordMatrix:
                    for word in grad:
                        self.ms[key][word] = self.beta_1 * self.ms[key][word] + (1 - self.beta_1) * grad[word]
                        self.vs[key][word] = self.beta_2 * self.vs[key][word] + (1 - self.beta_2) * np.square(grad[word])
                        self.theta[key][word] -= lr * self.ms[key][word] / np.sqrt(self.vs[key][word] + self.epsilon)
                else:
                    raise NameError("Cannot update theta")







