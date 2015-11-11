import numpy as np
import theano
import theano.tensor as T

np.random.seed(2)

word1 = np.array([0,0,0,1,0,0,0,0,0,0]).astype(theano.config.floatX)
word2 = np.array([0,0,1,0,0,0,0,0,0,0]).astype(theano.config.floatX)
word3 = np.array([0,1,0,0,0,0,0,0,0,0]).astype(theano.config.floatX)
word4 = np.array([1,0,0,0,0,0,0,0,0,0]).astype(theano.config.floatX)
words = [word1, word2, word3, word4]

seq1 = [words[i] for i in np.random.choice(3, 6)]
seq2 = [words[i] for i in np.random.choice(3, 6)]
seq3 = [words[i] for i in np.random.choice(3, 6)]
seqs = np.array([seq1, seq2]).transpose(1,0,2)

lexicon = np.zeros((10,10)).astype(theano.config.floatX)
np.fill_diagonal(lexicon, 1)
a, b, c, d, e, f, g, h, i, j = lexicon
seq1 = np.array([a, b, b, b, b, d])
seq2 = np.array([f, b, b, b, b, i])
seqs = np.array([seq1, seq2]).transpose(1, 0, 2)
# print '\ntest theano seqs: \n', seqs

input_size = len(seq1[1])
hidden_size = 4


U = theano.shared(value=np.random.randn(input_size,hidden_size), name='U')
V = np.random.randn(hidden_size,hidden_size)
W = np.random.randn(hidden_size,input_size)
b1 = np.random.normal(0, 0.1, hidden_size)
b2 = np.random.normal(0, 0.1, input_size)


# for single sequences
input_sequence = T.matrix("input_sequence", dtype=theano.config.floatX)
hidden_t = T.vector("hidden_t", dtype=theano.config.floatX)

def prediction(output_vector):
    prediction = T.argmax(output_vector, axis=1)
    return prediction

# func to compute hidden layer activation
def calc_hidden(input_t, hidden_t):
    return T.nnet.sigmoid(T.dot(input_t, U) + T.dot(hidden_t, V) + b1)

hidden_sequence, _ = theano.scan(calc_hidden, sequences=input_sequence, outputs_info=hidden_t)
output_sequence = T.nnet.softmax(T.dot(hidden_sequence, W) + b2)[:-1]
predictions_sg = prediction(output_sequence)
predictions_error_sg = T.eq(predictions_sg, prediction(input_sequence[1:]))
p_error_sg = T.mean(predictions_error_sg)
prediction_last_sg = 1 - predictions_error_sg[-1]
errors_sg = T.nnet.categorical_crossentropy(output_sequence, input_sequence[1:])
error_sg = T.mean(errors_sg)

hidden_init = np.zeros(hidden_size).astype(theano.config.floatX)
givens = {hidden_t : hidden_init}
error = theano.function([input_sequence], error_sg, givens=givens)
output = theano.function([input_sequence], output_sequence, givens=givens)
prediction_sg = theano.function([input_sequence], predictions_sg, givens=givens)
prediction_error_sg = theano.function([input_sequence], prediction_last_sg, givens=givens)


##################################################################
##################################################################
# for matrices 

input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)

def predictions(output_vector):
    prediction = T.argmax(output_vector, axis=2)
    return prediction

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
def calc_hiddens(input_t, hidden_t):
    return T.nnet.sigmoid(T.dot(input_t, U) + T.dot(hidden_t, V) + b1)

hiddens_t = T.matrix("hiddens_t", dtype=theano.config.floatX)

hidden_sequences, _ = theano.scan(calc_hiddens, sequences=input_sequences, outputs_info=hiddens_t)
output_sequences = softmax_tensor(T.dot(hidden_sequences, W) + b2)[:-1]
predictions_pl = predictions(output_sequences)
predictions_error_pl = T.eq(predictions_pl, predictions(input_sequences[1:]))
p_error_pl = T.mean(predictions_error_pl, axis=0)
prediction_last_pl = T.mean(1 - predictions_error_pl[-1])
errors_pl = T.nnet.categorical_crossentropy(output_sequences, input_sequences[1:])
error_pl = T.mean(errors_pl)


d1, d2, d2 = input_sequences.shape
hidden_inits = T.zeros(shape=(input_sequences.shape[1],hidden_size)).astype(theano.config.floatX)
givens = {hiddens_t : hidden_inits}
error_mat = theano.function([input_sequences], error_pl, givens=givens)
output_mat = theano.function([input_sequences], output_sequences, givens=givens)
prediction_pl = theano.function([input_sequences], predictions_pl, givens=givens)
prediction_error_pl = theano.function([input_sequences], prediction_last_pl, givens=givens)

##################################################################
##################################################################


##################################################################
### Computing and printing

output1 = output(seq1)
output2 = output(seq2)
output3 = output(seq3)
output_matrix = output_mat(seqs)

prediction1 = prediction_sg(seq1)
prediction2 = prediction_sg(seq2)
# prediction3 = prediction_sg(seq3)
predictions = prediction_pl(seqs)

prediction_error1 = prediction_error_sg(seq1)
prediction_error2 = prediction_error_sg(seq2)
# prediction_error3 = prediction_error_sg(seq3)
prediction_errors = prediction_error_pl(seqs)


######################################3
# Printing
print "input sequences:\n", seqs

print "\n\nprediction computed as separate sequences:\n"
print prediction1, prediction2 #, prediction3
print "\n\nprediction computed as matrix:\n"
print predictions

print "\n\nprediction error computed as separate sequences:\n"
print prediction_error1, prediction_error2 #, prediction_error3
print "\n\nprediction computed as matrix:\n"
print prediction_errors

