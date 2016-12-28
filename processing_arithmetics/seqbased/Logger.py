from keras.callbacks import Callback


# noinspection PyAttributeOutsideInit
class Logger(Callback):

    def __init__(self, N):
        """
        Print summary of current losses and accuracy
        every N epochs
        """
        self.N = N

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.prediction_error = []
        self.val_prediction_error = []
        self.i = 0
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:
            # give summary of current results
            print('Epoch %i' % epoch)
            # print('training loss: %f - val loss: %f - training mspe: %f - val mspe: %f'
            print('training loss: %f - val loss: %f'
                  % (logs.get('loss'), logs.get('val_loss')))
                 #    logs.get('mean_squared_prediction_error'),
                 #    logs.get('val_mean_squared_prediction_error')))
