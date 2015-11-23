import numpy as np
import matplotlib.pylab as plt

# params
damp_factor = 0.5
N = 20
n = 300

# init
network_state = (0.5-np.random.random(N))

# weight matrices
W_in = damp_factor*(0.5-np.random.random((N,N)))             # weightmatrix
W_out_back = (0.5-np.random.random(N))          # feed output back into network

train_seq = np.array([0.5*np.sin(float(x)/4) for x in xrange(n+1)])

# plot oscillations DR without input
states = []

for i in xrange(50):
    network_state = np.tanh(np.dot(network_state, W_in))
    states.append(network_state)

x = xrange(50)
s = np.array(states)

plt.figure(1)
for i in xrange(20):
    plt.subplot(4, 5, i)
    plt.plot(x, s[:,i])

plt.show()


network_state = np.zeros(N)     # start from 0 state
states = []

# loop and train
for i in xrange(n):
    network_state = np.tanh(np.dot(network_state, W_in) + np.dot(train_seq[i],W_out_back))
    states.append(network_state)

# plot DR 100-105
s = np.array(states)
x = xrange(50)
plt.figure(1)
for i in xrange(20):
    plt.subplot(4, 5, i)
    plt.plot(x, s[100:150,i])

plt.show()

t = np.array(train_seq[-200:]).reshape(200,1)

# compute output matrix
W_out = np.dot(np.linalg.pinv(s[-200:]), t).reshape(20,)

# test phase, start from last network state
output = []
states = []
for i in xrange(50):
    out = np.dot(network_state, W_out)
    network_state = np.tanh(np.dot(network_state, W_in) + np.dot(out, W_out_back))
    output.append(out)
    states.append(network_state)

s = np.array(states)

# Plot internal states after training
# plt.figure(1)
# for i in xrange(20):
#     plt.subplot(4, 5, i)
#     plt.plot(x, s[:,i])
# 
# plt.show()

# plot network behaviour after training
x_range = xrange(50)
plt.plot(x_range, output, label="Network output")
plt.plot(x_range, [0.5*np.sin(float(k+n)/4) for k in x_range], label="target")
plt.legend()
plt.show()
