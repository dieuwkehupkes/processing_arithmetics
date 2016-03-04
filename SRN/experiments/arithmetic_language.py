from SRN_Theano import SRN
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from collections import OrderedDict

N_training = 3          # nr of training examples

network = SRN(2, 2, 0.2)

# generate initial word embeddings for numbers
# -10 tot 10 and + and -
embeddings = np.random.normal(1, size=(124,2))

# create shared variables to store word embeddings
names = [str(i) for i in xrange(-19, 20)] + ['+', '-', '=', '(',')']
words = OrderedDict()
values = dict()
for i in xrange(len(names)):
    words[names[i]] = theano.shared(value=embeddings[i].astype(theano.config.floatX), name=names[i])
    try:
       values[names[i]] = int(names[i])
    except ValueError:
        pass

# words['b1'] = 'whatever'

training = []
# generate some training examples with 2 numbers
for i in xrange(N_training):
    x1, x2 = np.random.randint(-10, 11, size=2)   # generate two numbers
    op = np.random.choice(['+', '-'])
    if op == '+':
        outcome = x1 + x2
    elif op == '-':
        outcome = x1 - x2

    training_example = np.array([words[str(x1)],
         words[op],
         words[str(x2)],
         words['='],
         words[str(outcome)]]) 
    
    training.append(training_example)

print(training)

for batch in training:
    print batch
    for embedding in batch:
        print embedding
exit()

network.create_model(training, words)
# network.generate_update_function()

