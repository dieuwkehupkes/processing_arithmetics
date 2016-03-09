import numpy as np
import theano
import theano.tensor as T

class SRN():

    def __init__(self):

        self.embeddings = theano.shared(
            value = np.random.uniform(-1, 1, (10,8)),
            name = 'embeddings'
        )

        self.W = theano.shared(
                value = np.identity(8).astype(theano.config.floatX),
                name = 'W'
        )

        self.V = theano.shared(
                value = np.zeros((8, 8)).astype(theano.config.floatX),
                name = 'W'
        )

        # Now generate the function that we use to compute the closest

        # declare input sequence, transpose to use scan
        input_sequences = T.tensor3("input_sequences", dtype=theano.config.floatX)
        input_sequences_map = T.dot(input_sequences, self.embeddings)
        input_sequences_map_transpose = input_sequences_map.transpose(1,0,2)

        # compute input map from input

        def calc_hidden(input_vector, prev_vector):
            # use scan to compute output sequence from
            # input sequence
            return T.nnet.sigmoid(T.dot(input_vector, self.W) + T.dot(prev_vector, self.V))

        hidden_t = T.matrix('init_output', dtype=theano.config.floatX)

        # apply scan to compute output sequences
        hidden_sequences, _ = theano.scan(calc_hidden, sequences=input_sequences_map_transpose, outputs_info=hidden_t)

        output_sequences = T.nnet.sigmoid(T.dot(hidden_sequences, self.W))[-2]
        # take the last element to get a vector with vectors

        # now compute distances and closest vector analog to numpy
        theano_distances = T.sqrt(T.sum(T.sqr(self.embeddings[:, None] - output_sequences), axis=2))
        theano_closest = T.argmin(theano_distances, axis=0)

        givens = {hidden_t: T.zeros((2,8)).astype(theano.config.floatX)}

        # function to inspect outputs
        self.generate_outputs = theano.function([input_sequences], output_sequences, givens=givens)

        # function to get the closest output vectors
        self.get_closests = theano.function([input_sequences], theano_closest, givens=givens)

        return


# test if this works

# create target matrix and give names to elements
lexicon = np.identity(10).astype(theano.config.floatX)
a, b, c, d, e, f, g, h, i, j = lexicon

input_seq = np.array([[a,b,d],[f,b,i]])

network = SRN()
network.generate_prediction_function()
print network.generate_outputs(input_seq)
print network.get_closests(input_seq)


