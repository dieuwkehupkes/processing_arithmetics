import numpy as np
import theano
import theano.tensor as T

np.random.seed(0)

U = theano.shared(value=np.random.randn(3,2), name='U')
V = np.random.randn(2,2)
W = np.random.randn(2,3)

seq1 = np.random.randn(5, 3)
seq2 = np.random.randn(5, 3)
seq3 = np.random.randn(5, 3)
seqs = np.array([seq1, seq2, seq3]).transpose(1,0,2)

input_sequence = T.matrix("input_sequence", dtype=theano.config.floatX)
input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)

def softmax_tensor(t):
    """
    Compute softmax from tensor
    """
    d0, d1, d2 = t.shape
    reshaped = T.reshape(t, (d0*d1, d2))
    sm = T.nnet.softmax(reshaped)
    sm_reshaped = T.reshape(sm, newshape=t.shape)
    return sm_reshaped

# func to compute hidden layer activation
def calc_hidden(input_t, hidden_t):
    return T.nnet.sigmoid(T.dot(input_t, U)) + T.dot(hidden_t, V)

# func to compute hidden layer activation
def calc_hiddens(inputs_t, hiddens_t):
    return T.nnet.sigmoid(T.dot(inputs_t, U)) + T.dot(hiddens_t, V)

hidden_t = T.vector("hidden_t", dtype=theano.config.floatX)
hiddens_t = T.matrix("hiddens_t", dtype=theano.config.floatX)

hidden_sequence, _ = theano.scan(calc_hidden, sequences=input_sequence, outputs_info=hidden_t)
hidden_sequences, _ = theano.scan(calc_hiddens, sequences=input_sequences, outputs_info=hiddens_t)
# hidden_sequences = hidden_sequences_T.transpose(1,0,2)

output_sequence = T.nnet.softmax(T.dot(hidden_sequence, W))[:-1]
output_sequences = softmax_tensor(T.dot(hidden_sequences, W))[:-1]

errors_sg = T.nnet.categorical_crossentropy(output_sequence, input_sequence[1:])
errors_pl = T.nnet.categorical_crossentropy(output_sequences, input_sequences[1:])
error_sg = T.mean(errors_sg)
error_pl = T.mean(errors_pl, axis=0)

hidden_init = np.zeros(2).astype(theano.config.floatX)
givens = {hidden_t : hidden_init}
hidden = theano.function([input_sequence], error_sg, givens=givens)

d1, d2, d2 = input_sequences.shape
hidden_inits = np.zeros(shape=(3,2)).astype(theano.config.floatX)
hidden_inits = np.zeros(shape=(3,2)).astype(theano.config.floatX)
givens = {hiddens_t : hidden_inits}
hidden_mat = theano.function([input_sequences], error_pl, givens=givens)

error1= hidden(seq1)
error2= hidden(seq2)
error3= hidden(seq3)

print "computed as separate sequences:\n" 
print "H", error1
print "\nH", error2
print "\nH", error3

hidden_matrix = hidden_mat(seqs)

print "computed as matrix"
print hidden_matrix
# print hidden_matrix2

