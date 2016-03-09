import numpy as np
import theano
import theano.tensor as T

np.random.seed(10)
# create target matrix and give names to elements
lexicon = np.identity(10).astype(theano.config.floatX)
a, b, c, d, e, f, g, h, i, j = lexicon

embeddings_values = np.random.uniform(-1, 1, (10,8))

input_seq = np.array([[a,b,d],[f,b,i]])

# definieer shared variable for embeddings
embeddings = theano.shared(
        value = embeddings_values,
        name = 'embeddings'
)

# definieer some dummy matrices to compute output from input
W = theano.shared(
        value = np.identity(8).astype(theano.config.floatX),
        name = 'W'
)

V = theano.shared(
        value = np.zeros((8, 8)).astype(theano.config.floatX),
        name = 'W') 

# declare input sequence, transpose to use scan
input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)
input_sequences_map = T.dot(input_sequences, embeddings)
input_sequences_transpose = input_sequences_map.transpose(1,0,2)

# compute input map from input

def calc_hidden(input_vector, prev_vector):
    # use scan to compute output sequence from
    # input sequence
    return T.nnet.sigmoid(T.dot(input_vector, W) + T.dot(prev_vector, V))

# declare first hidden state
hidden_t = T.matrix('init_output', dtype=theano.config.floatX)

# apply scan to compute output sequences
hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences_transpose, outputs_info=hidden_t)      # tensor

# compute output sequences by applying weight matrix and activation function
# take 1 but last output as prediction
output_sequences = T.nnet.sigmoid(T.dot(hidden_sequences, W))[-2]   # matrix

# now compute distances and closest vector analog to numpy
theano_distances = T.sqrt(T.sum(T.sqr(embeddings[:, None, :] - output_sequences), axis=2))
theano_closest = T.argmin(theano_distances, axis=0)

givens = {hidden_t: T.zeros((2,8)).astype(theano.config.floatX)}

# function to inspect outputs
generate_outputs = theano.function([input_sequences], output_sequences, givens=givens)

# function to get the closest output vectors
get_closests = theano.function([input_sequences], theano_closest, givens=givens)

# now lets execute the function, we apply it to the sequences
# a, a, a and b, e, e
# as the matrix V is empty (and the previous activation thus doesn't get copied) and
# the matrix W is the identity matrix, this should give us the same output
# as in the numpy case

print get_closests(input_seq)


