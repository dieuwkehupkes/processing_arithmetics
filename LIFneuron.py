import numpy as np
import matplotlib.pylab as plt

I1 = 1.5
I2 = 2.2
R = 1
tau = 0.1
treshold = 1.3

timestep = 0.01
simLength = 80

deriv = lambda v, R, tau, I: (-v + R*I)/tau

t1, t2 = [0.0], [0.0]
v1, v2 = [0.0], [0.0]
v1_value, v2_value = 0, 0
t_value = 0

for i in xrange(simLength):

    # compute next values
    v1_deriv = timestep*deriv(v1_value, R, tau, I1)
    v2_deriv = timestep*deriv(v2_value, R, tau, I2)
    v1_value = v1_value + v1_deriv
    v2_value = v2_value + v2_deriv
    t_value += timestep

    # add values to list
    v1.append(v1_value)
    v2.append(v2_value)
    t1.append(t_value)
    t2.append(t_value)

    # reset values if passing treshold
    if v1_value > treshold:
        v1_value = 0
        t1.append(t_value)
        v1.append(v1_value)
    if v2_value > treshold:
        v2_value = 0
        v2.append(v2_value)
        t2.append(t_value)

plt.axhline(y=treshold, linestyle='--', color='r')
plt.ylim(-0.2, 1.7)
plt.xlim(0.0, 0.52)
plt.plot(t1, v1, label="I=1.5")
plt.plot(t2, v2, label="I=2.0")
plt.yticks([])
plt.xticks([])
plt.ylabel('V(t)')
plt.xlabel('t')
plt.show()
