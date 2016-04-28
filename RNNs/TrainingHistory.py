from keras.callbacks import Callback


# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.prediction_error = []
        self.val_prediction_error = []
        self.i = 0
    
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.prediction_error.append(logs.get('mean_squared_prediction_error'))
        self.val_prediction_error.append(logs.get('mean_squared_prediction_error'))
        self.i += 1
