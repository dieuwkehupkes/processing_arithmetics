import numpy as np
import matplotlib.pyplot as plt

def compute_sin(x):
    return 0.5 * np.sin(0.25*x)

def initialize_matrices(n,l,alpha):
    
    assert alpha>0 and alpha<1, 'Error in alpha value'
    
    try:
        # initialize randomly
        w = 0.5-np.random.rand(n,n)
        w_mask = np.random.rand(n,n) >= 0.44
        w = w*w_mask
        w_back = np.random.rand(n,l)

        # get eigenvalues
        eigs = np.linalg.eigvalsh(w)
        max_eig = np.max(eigs)

        assert max_eig!=0, 'Try again'

        w = w*(1/max_eig)
    
        # scale to alpha
        w = alpha*w
    except AssertionError:
            w,w_back = initialize_matrices(n,l,alpha)

    return w,w_back

if __name__=='__main__':
    # set seed
    #numpy.random.seed(1234)

    #compute_sin(1)
    n = 20
    l = 1
    iters = 300
    t_0 = 100
    alpha = .8
    m = np.zeros((iters-t_0,n))
    t = np.zeros((iters-t_0,l))


    w,w_back = initialize_matrices(n,l,alpha)
    #---
    #w = alpha*(0.5 - np.random.rand(n,n))
    #w_back = 0.5 - np.random.rand(n,l)

    network_state = (0.5-np.random.random(n))
    states = []

    for i in xrange(20):
        network_state = np.tanh(np.dot(network_state, w))
        states.append(network_state)

    x = xrange(20)
    s = np.array(states)

    plt.figure(1)
    for i in xrange(20):
        plt.subplot(4, 5, i)
        plt.plot(x, s[:,i])

    plt.show()
    #---

    x_n = np.zeros((n,1))
    d_n = 0

    # sampling stage
    for i in range(iters):
        # update xn
        x_n = np.tanh(w.dot(x_n)+w_back.dot(d_n))
        d_n = compute_sin(i)
        
        if i > t_0:
            m[i-t_0,:] = x_n[:,0]
            #t[i-t_0,:] = np.arctanh(d_n)
            t[i-t_0,:] = d_n

    # weight update stage
    m_inv = np.linalg.pinv(m)
    w_out = m_inv.dot(t)

    print "average w_out ", np.mean(w_out)
    print "variance w_out ", np.var(w_out)

    # plot x_n
    n_plot = 50
    x_init_plot = 0

    for i in range(n):
        plt.plot(range(n_plot),m[x_init_plot:x_init_plot+n_plot,i])
        plt.ylim((-1,1))
    plt.show()

    mses = []
    n_additional = 50
    true_values = []
    pred_values = []
    y_n = w_out.T.dot(x_n) # prediction y(49)
    for i in range(n_additional):
        j = iters+i
        x_n = np.tanh(w.dot(x_n)+w_back.dot(y_n))
        y_n = w_out.T.dot(x_n)
        #y_n = w_out.T.dot(x_n)
        #y_n = np.tanh(w_out.T.dot(x_n))
        #y_n = w_out.T.dot(x_n)
        true_val = compute_sin(j)
        mses.append((true_val-y_n[0][0])**2)
        true_values.append(true_val)
        pred_values.append(y_n[0][0])

    plt.plot(range(n_additional),true_values,color='green',label='true')
    plt.plot(range(n_additional),pred_values,color='red',label='pred')
    plt.ylim((-.5,.5))
    plt.legend()
    plt.show()
