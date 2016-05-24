from keras.callbacks import Callback
import numpy as np


# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    Track different aspects of the network and network performance
    during training.
    """
    def __init__(self, metrics, recurrent_id, param_id=1):
        self.recurrent_id = recurrent_id
        self.param_id = param_id
        self.metrics = metrics

    def on_train_begin(self, logs={}):
        self.losses = []                # track training loss
        self.val_losses = []            # track validation loss
        self.prediction_error = []      # track training prediction error
        self.metrics_train = dict([(metric, []) for metric in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(metric, []) for metric in self.metrics])         # treck metrics on validation set
        self.esp = []                   # track esp recurrent layer
        self.i = 0                      # track epoch nr

    def on_epoch_end(self, epoch, logs={}):
        """
        Append tracking objects to their respective lists
        """
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        for metric in self.metrics:
            self.metrics_train[metric].append(logs.get(metric))
            self.metrics_val[metric].append(logs.get('val_'+metric))
        self.i += 1

        # compute esp
        recurrent_weights = self.model.layers[self.recurrent_id].get_weights()[self.param_id]
        spec = np.max(np.absolute(np.linalg.eig(recurrent_weights)[0]))
        self.esp.append(spec)

