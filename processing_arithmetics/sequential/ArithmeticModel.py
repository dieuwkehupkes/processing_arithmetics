from keras.models import Model
import copy

class ArithmeticModel(Model):
    """
    A keras model with an extra attribute,
    that describes the the mapping of the words
    in the arithmetic language to integers that
    the model assumes.
    """
    def __init__(self, inputs, outputs, dmap, name=None):
        # call __init__ of superclass
        super(ArithmeticModel, self).__init__(inputs, outputs, name)
        self.dmap = dmap

    def get_config(self):
        """
        Retuns the model configuration as a
        dictionary.
        """
        config = super(ArithmeticModel, self).get_config()
        config['dmap'] = self.dmap
        return copy.deepcopy(config)
