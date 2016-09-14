from generate_training_data import string_to_vec
import pickle

# generate training sequences for the network

dmap =  'models/GRU2.dmap'
maxlen = 57
expressions = ['( ( 1 + 5 ) + 7 )',
               '( ( 1 - 5 ) + 7 )',
               '( ( 1 + 5 ) - 7 )',
               '( ( 1 - 5 ) - 7 )'
               ]

if __name__ == '__main__':
    dmap = pickle.load(open(dmap, 'rb'))
    answers = [(eval(expression)) for expression in expressions]
    seqs = string_to_vec(expressions=expressions, dmap=dmap, pad_to=maxlen)
    pickle.dump([seqs, answers], open('test_input_sequences', 'wb'))
