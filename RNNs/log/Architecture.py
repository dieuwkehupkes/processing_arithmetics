from keras.models import Graph
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM
from TrainingHistory import TrainingHistory

class Architecture():
    """
    Class description.
    """
    def __init__(size_hidden, size_compare):
        self.model = Graph()
