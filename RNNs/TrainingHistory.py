from keras.callbacks import Callback
import numpy as np


# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    Track different aspects of the network and network performance
    during training.
    """

    def on_train_begin(self, logs={}):
        self.losses = []                # track training loss
        self.val_losses = []            # track validation loss
        self.prediction_error = []      # track training prediction error
        self.val_prediction_error = []  # track validation rpediction error
        self.esp = []                   # track esp recurrent layer
        self.i = 0                      # track epoch nr

    def on_epoch_end(self, epoch, logs={}):
        """
        Append tracking objects to their respective lists
        """
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.prediction_error.append(logs.get('mean_squared_prediction_error'))
        self.val_prediction_error.append(logs.get('val_mean_squared_prediction_error'))
        self.i += 1

        # compute esp
        recurrent_weights = self.model.layers[2].get_weights()[1]
        spec = np.max(np.absolute(np.linalg.eig(recurrent_weights)[0]))
        self.esp.append(spec)

