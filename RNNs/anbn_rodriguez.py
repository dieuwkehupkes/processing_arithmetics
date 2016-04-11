# run idealised solution Rodriguez and visualise outcome

import numpy as np
from SRN import SRN
import matplotlib.pylab as plt


# alphabet
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
E = np.array([0, 0, 1])

s1 = np.array([a,b,E])
s2 = np.array([a,a,b,b,E])
s3 = np.array([a,a,a,b,b,b,E])
s4 = np.array([a,a,a,a,b,b,b,b,E])
s5 = np.array([a,a,a,a,a,b,b,b,b,b,E])
s6 = np.array([a,a,b,a,b,b,E])

# generate network
srn = SRN(input_size=3, hidden_size=3, sigma_init=0.01, learning_rate=0.5)

# set weights according to rodriguez
srn.U = np.array([[0.5, -5, -5], [-5, -1, 5],[-5, -5, -5]])
srn.V = np.array([[0.5, 2, 0], [0, 2, 0], [0, 0, 0]])
srn.b1 = np.array([0,0,0])
srn.b2 = np.array([0,0,0])

# generate dynamics
srn.generate_update_function()

# vars to store values for visualisation
project_input = []
project_hidden = [[0,0,0]]
project = []
sumh = [[0,0,0]]
squash = [[0,0,0]]

sequence = s4
l = len(sequence)

for i in xrange(l):
    # raw_input()
    new_input = sequence[i]
    project_input.append(srn.project_input(new_input))
    project_hidden.append(srn.project_hidden())
    sumh.append(srn.sumh(new_input))
    squash.append(srn.squash(new_input))
    print(srn.squash(new_input))
    srn.forward_pass(new_input)
    # raw_input()

# create plotting list
x_sum = np.array([sumh[i][0] for i in xrange(l)])
x_squash = np.array([squash[i][0] for i in xrange(l)])
y_sum = np.array([sumh[i][1] for i in xrange(l)])
y_squash = np.array([squash[i][1] for i in xrange(l)])
 
plt.figure()

# plot effect of summing
plt.quiver(x_squash[:-1], y_squash[:-1], x_sum[1:]-x_squash[:-1], y_sum[1:]-y_squash[:-1], scale_units='xy', angles='xy', scale=1, color='blue')

# plot effect of squashing
plt.quiver(x_sum[1:], y_sum[1:], x_squash[1:]-x_sum[1:], y_squash[1:]-y_sum[1:], scale_units='xy', angles='xy', scale=1, color='red')
plt.plot(x_squash, y_squash,'k')
plt.plot(x_squash, y_squash,'bo')
plt.ylim((-3,1))
plt.xlim((-3,1))
plt.show()
