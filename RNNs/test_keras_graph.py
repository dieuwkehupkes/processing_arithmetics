from keras.models import Sequential, Graph
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM
from keras.utils.visualize_util import plot
from auxiliary_functions import pad
import numpy as np
import theano
import matplotlib.pyplot as plt
from TrainingHistory import TrainingHistory

# GENERATE TEST/TRAINING DATA

# Lexicon
a, b, c, d, e, f, g, h, i, j = np.arange(10)

# generate sequences, padd them to same length
seq1, l1 = pad(np.array([a,b,d]), 6), 'abd'
seq2, l2 = pad(np.array([f,b,i]), 6), 'fbi'
seq3, l3 = pad(np.array([a, b, b, d]), 6), 'abbd'
seq4, l4 = pad(np.array([f, b, b, i]), 6), 'fbbi'
seq5, l5 = pad(np.array([a, b, b, b, d]), 6), 'abbbd'
seq6, l6 = pad(np.array([f, b, b, b, i]), 6), 'fbbbi'
seq7, l7 = pad(np.array([a, b, b, b, b, d]), 6), 'abbbbd'
seq8, l8 = pad(np.array([f, b, b, b, b, i]), 6), 'fbbbbi'

embeddings = np.random.uniform(-0.5, 0.5, (10, 8)).astype(theano.config.floatX)

training_sequence1, targets1 = np.array([seq1[:-1], seq2[:-1]]), np.array([seq1[-1], seq2[-1]])
training_sequence2, targets2 = np.array([seq3[:-1], seq4[:-1]]), np.array([seq3[-1], seq4[-1]])
training_sequence3, targets3 = np.array([seq5[:-1], seq6[:-1]]), np.array([seq5[-1], seq6[-1]])
training_sequence4, targets4 = np.array([seq7[:-1], seq8[:-1]]), np.array([seq7[-1], seq8[-1]])

training_options = {'1': (training_sequence1, seq1, seq2, l1, l2),
                    '2': (training_sequence2, seq3, seq4, l3, l4),
                    '3': (training_sequence3, seq5, seq6, l5, l6),
                    '4': (training_sequence4, seq7, seq8, l7, l8)
                    }


# GENERATE NETWORK
input_size = 10
hidden_size = 8
seq_length = 6
classes = 10

model = Graph()
model.add_input(name='input', input_shape=(1,), dtype='int')
# create layer that maps integer between 0-10 to vector of length 8
# generates outputs of shape (batch_dim, 5, 8)
model.add_node(Embedding(10, 8, input_length=5), name='embeddings', input='input')

# Add RNN
model.add_node(GRU(8), name='recurrent1', input='embeddings')

# add softmax layer
model.add_node(Dense(10, activation='softmax'), name='output', input='recurrent1', create_output=True)

# compile model
print('compile model')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# print model.layers
# exit()
# 
# train model
print('train model')
history = TrainingHistory()
model.fit({'input':training_sequence4, 'output':targets1}, batch_size=2, nb_epoch=3000, verbose=0, shuffle=True, callbacks=[history])
# 
plt.plot(history.losses, label='loss')
plt.show()
# 
# outputs = model.predict_classes(training_sequence4, batch_size=2)
# print(outputs)
