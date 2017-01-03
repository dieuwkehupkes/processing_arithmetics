from keras.layers import SimpleRNN, GRU, LSTM
from processing_arithmetics.seqbased.architectures import Training, ScalarPrediction, ComparisonTraining, DiagnosticClassifier, Seq2Seq

# file with transformation functions used by argparse

def get_architecture(architecture):
    arch_dict = {'ScalarPrediction':ScalarPrediction, 'ComparisonTraining':ComparisonTraining, 'DiagnosticClassifier':DiagnosticClassifier, 'DC':DiagnosticClassifier, 'Seq2Seq':Seq2Seq}
    return arch_dict[architecture]

def get_hidden_layer(hl_name):
    hl_dict = {'SimpleRNN': SimpleRNN, 'SRN': SimpleRNN, 'GRU': GRU, 'LSTM': LSTM}
    return hl_dict[hl_name]

def max_length(N):
    """
    Compute length of arithmetic expression
    with N numeric leaves
    :param N: number of numeric leaves of expression
    """
    l = 4*N-3
    return l
