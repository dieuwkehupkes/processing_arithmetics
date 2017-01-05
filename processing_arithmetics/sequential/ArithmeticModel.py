from keras.models import Model
from collections import OrderedDict
import numpy as np

class ArithmeticModel(Model):
    """
    A keras model with an extra attribute,
    that describes the the mapping of the words
    in the arithmetic language to integers that
    the model assumes.
    """
    def __init__(self, input, output, dmap, name=None):
        # call __init__ of superclass
        super(ArithmeticModel, self).__init__(input, output, name)
        self.dmap = dmap
